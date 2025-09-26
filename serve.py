# serve.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer

app = FastAPI()

MODEL_DIR = "models"

# load artifacts
clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib"))

# models for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
img_model = models.resnet50(pretrained=True)
img_model.fc = nn.Identity()
img_model = img_model.to(device)
img_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_image_embedding(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    x = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = img_model(x).cpu().numpy().squeeze()
    return feat

def top_tags_from_text(text, top_k=5):
    # get tfidf feature scores for this text, return top token terms
    vec = tfidf.transform([text])
    indices = np.argsort(-vec.toarray()[0])[:top_k]
    features = np.array(tfidf.get_feature_names_out())
    return features[indices].tolist()

@app.post("/predict")
async def predict(image: UploadFile = File(...), title: str = Form(""), description: str = Form(""), top_k_tags: int = Form(5)):
    text = (title or "") + " " + (description or "")
    # text embedding
    text_emb = text_model.encode(text, convert_to_numpy=True)

    # image embedding
    file_bytes = await image.read()
    try:
        img_emb = extract_image_embedding(file_bytes)
    except Exception as e:
        img_emb = np.zeros(2048, dtype=np.float32)

    X = np.hstack([text_emb, img_emb]).reshape(1, -1)
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[0]
    pred_idx = int(probs.argmax())
    pred_label = le.inverse_transform([pred_idx])[0]
    top_tags = top_tags_from_text(text, top_k=top_k_tags)

    # include top-n predicted category probabilities if you want:
    top_n = 3
    top_inds = probs.argsort()[::-1][:top_n]
    top_categories = [{"category": le.inverse_transform([int(i)])[0], "prob": float(probs[int(i)])} for i in top_inds]

    return {"category": pred_label, "score": float(probs[pred_idx]), "top_categories": top_categories, "tags": top_tags}
