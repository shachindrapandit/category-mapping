# train_data.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# paths
DATA_CSV = "/home/spandit/category-mapping-data/data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load data
df = pd.read_csv(DATA_CSV)
df['text'] = (df['title'].fillna('') + " " + df['description'].fillna('')).astype(str)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# text embedder
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast

# image model: resnet50, remove final fc
img_model = models.resnet50(pretrained=True)
img_model.fc = nn.Identity()
img_model = img_model.to(device)
img_model.eval()

# image transforms
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# compute features
text_embs = []
image_embs = []
failed_images = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    text_embs.append(text_model.encode(row['text'], convert_to_numpy=True))

    img_path = row['image_path']
    try:
        img = Image.open(img_path).convert('RGB')
        x = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = img_model(x).cpu().numpy().squeeze()
        image_embs.append(feat)
    except Exception as e:
        failed_images += 1
        # fallback: zero vector if image missing/corrupt
        image_embs.append(np.zeros(2048, dtype=np.float32))

text_embs = np.vstack(text_embs)
image_embs = np.vstack(image_embs)

# concat features
X = np.hstack([text_embs, image_embs])

# labels
le = LabelEncoder()
y = le.fit_transform(df['category'].astype(str))

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# classifier (logistic regression multiclass)
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
clf.fit(X_scaled, y)

# TF-IDF vectorizer for tag extraction
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
tfidf.fit(df['text'].values)

# save artifacts
joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))

# note: text_model and image model are reloaded by the server (they are pretrained)
print(f"Done. Failed images: {failed_images}")
