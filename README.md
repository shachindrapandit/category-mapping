About POC: Automated Automated Product Tagging & Categorization

🎯 Objective:
Automatically assign relevant tags (e.g., “organic”, “wireless”, “waterproof”) and map products to appropriate categories (e.g., “Electronics > Headphones > Wireless”) using product metadata such as titles, descriptions, specifications, and images.

📦 Input Data:

Product Title

Product Description

Specifications (structured key-values)

Images (optional, depending on scope)

Existing tags/categories (for training, if available)

🚀 Use Cases:

Product ingestion from suppliers or sellers

Tag suggestions for search filtering

Category mapping for catalog normalization

SEO tag enrichment

Duplicate detection or misclassification correction

🛠️ Technical Architecture:

Data Ingestion Layer:

Input via CSV, API, or database

Normalize fields (e.g., strip HTML, clean descriptions)

Feature Extraction:

Text: Title + Description + Specs (concatenated)

Optional: Image embedding via vision models (CLIP or BLIP)

AI Model Layer:

📌 Tagging (multi-label classification):

Model: BERT, RoBERTa, or OpenAI embeddings + classifier

Output: Set of tags with confidence scores

📌 Categorization (multi-class classification):

Model: Fine-tuned LLM or XGBoost with embeddings

Output: Best-matching category path

Post-Processing:

Thresholding (confidence cutoffs)

Deduplication or conflict resolution

Logging/feedback loop

Output:

Tags: ["wireless", "bluetooth", "noise-cancelling"]

Category: Electronics > Audio > Headphones > Wireless

🧪 Model Training (optional for POC):

Train/Test split if labeled data exists

Fine-tune on internal product data

Augment with public datasets (e.g., Amazon Product Data, Rakuten)

📊 Evaluation Metrics:

For tagging:

Precision / Recall / F1 (multi-label)

For categorization:

Accuracy / Top-k Accuracy / Confusion Matrix

Human-in-the-loop: Have SMEs review a subset of predictions

⚙️ Tools & Technologies:

Task	Suggested Tools
Embeddings	OpenAI, Sentence-BERT, Cohere, HuggingFace models
Classification	scikit-learn, XGBoost, Transformers, LLMs (GPT-4)
Vector Search	Pinecone, Weaviate, FAISS (for semantic matching)
Backend/API	Python (FastAPI), Node.js (NestJS), or LangChain
UI (if needed)	Streamlit, Next.js, or Retool

🧭 Sample Workflow (POC):

Upload sample product dataset (e.g., 5,000 SKUs)

Run inference for tags and category prediction

Visualize results with confidence scores

Review and annotate errors (if feedback loop is in scope)

(Optional) Retrain and re-test with corrected labels

💡 POC Success Criteria:

≥85% accuracy for category prediction in top-3

≥75% F1-score for tag assignment

Ability to process 1,000+ products in <10 minutes

Easy API or batch export support

🧱 Possible Datasets for Bootstrapping:

Amazon Product Dataset (from UCSD or Kaggle)

WDC Product Data Corpus

Rakuten Product Classification Dataset (multilingual)

Open Product Data (OpenFoodFacts, GS1, etc.)

🔒 Optional Advanced Enhancements:

Multimodal tagging (combine text + image)

Active learning: prioritize edge cases for review

Language detection & multilingual support

Continuous fine-tuning with user feedback


POC Steps - 

Step-by-step POC plan (exact steps you can run locally)
0) WSL: bootstrap environment (commands)

Open WSL terminal (Ubuntu) and run:

# create a project dir
mkdir ~/prod-tagging-poc && cd ~/prod-tagging-poc

# create python venv
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip
pip install -U pip

# install core deps
pip install torch torchvision sentence-transformers scikit-learn pandas numpy pillow joblib fastapi uvicorn tqdm


(If you have a CUDA-capable GPU and want to use it, install the appropriate torch wheel for your CUDA version — follow PyTorch instructions — but CPU-only works for a small POC.)

1) Dataset format

Prepare a CSV data/products.csv with these columns:

id,image_path,title,description,category
1,images/sku123.jpg,"Red cotton t-shirt","100% cotton, round neck",Clothing>Men>T-Shirts
2,images/sku456.jpg,"Bluetooth earbuds","Wireless, noise cancelling",Electronics>Audio>Earbuds
...


Place the images under images/.

2) Minimal training approach (text+image embeddings → classifier)

Create train.py (below I give complete code). It:

loads CSV,

computes text embeddings with SentenceTransformer (all-MiniLM-L6-v2),

computes image features with pretrained ResNet50 (avgpool output),

concatenates features,

trains a LogisticRegression (or lightweight MLP) classifier,

fits a TF-IDF vectorizer for tag extraction,

saves artifacts (joblib + models).

train.py (paste into your project)

# train.py
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
DATA_CSV = "data/products.csv"
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


Run:

python train.py

3) Serve predictions (FastAPI)

Create serve.py — load back the models, accept multipart upload or URL + text, return category + top tags.

serve.py

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


Run server:

uvicorn serve:app --reload --host 0.0.0.0 --port 8000


Test it with curl (example):

curl -X POST "http://localhost:8000/predict" \
  -F "image=@images/sku123.jpg" \
  -F "title=Red cotton t-shirt" \
  -F "description=100% cotton, round neck" \
  -F "top_k_tags=4"

4) NestJS (or Next.js) integration: simple client

From your Node/NestJS service call the FastAPI endpoint; example with axios and form-data:

// node-client.ts
import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

async function send() {
  const form = new FormData();
  form.append('image', fs.createReadStream('images/sku123.jpg'));
  form.append('title', 'Red cotton t-shirt');
  form.append('description', '100% cotton, round neck');

  const res = await axios.post('http://localhost:8000/predict', form, {
    headers: form.getHeaders(),
  });

  console.log(res.data);
}
send();

Tag generation strategy (better-than-naive)

Use TF-IDF to extract candidate tags from title+description (we used that).

Use noun-phrase extraction (spaCy) or RAKE to get multiword tags.

Deduplicate & map tags to canonical taxonomy strings (normalization).

For ambiguous products: present top N suggestions to a human-in-the-loop review UI; store feedback and retrain periodically.

Evaluation & metrics (what to monitor on POC)

Classification: Accuracy, Macro F1, Precision@K (if multi-label or hierarchical).

Tagging: Precision@k (are top-k tags useful), human acceptance rate.

Data quality: label noise, class imbalance (use class_weight or resampling).

Inference latency & RAM (for production later).

Labeling / Human-in-the-loop

Use Label Studio (self-host) to create labeled categories and confirm tags.

Provide reviewers a UI: show model predictions + top tags, allow accept/edit → store edits to retrain.

Scale & production considerations (next steps)

Move feature extraction to GPU instances (if needed).

Replace sklearn classifier with a small neural head and finetune on your dataset.

Use FAISS or Milvus to store embedding index for similarity search.

Use batching and a model server (BentoML / TorchServe / Triton).

Add monitoring: logging inference inputs/outputs (beware PII), model drift alerts.

Quick checklist (what to do next)

 Prepare a small labeled dataset (200–2000 examples) across your main categories.

 Run the training script and validate predictions locally.

 Build a simple review UI (NextJS / React) to accept/reject tags.

 Add annotation & incremental retraining loop.

 If satisfied, prototype vector DB + CLIP-based retriever for improved multi-modal semantics.

If you want, I can:

generate the full project skeleton (file tree + files) as a zip you can download, or

give a follow-up that swaps ResNet for CLIP (joint embedding) and shows how to fine-tune, or

produce a minimal Next.js React admin page to review predicted tags and confirm them (includes code).

Which of those would help you most next?


Steps to run the application - 

1. new wsl terminal: (keep it running)  uvicorn serve:app --reload --host 0.0.0.0 --port 8000

2. new wsl terminal: (keep it running) npm run build and npm run start:dev

3. new wsl terminal: curl -X POST http://localhost:3000/tagging/predict \
  -F "image=@/home/spandit/category-mapping-images/image-1.jpg" \
  -F "title=Red cotton t-shirt" \
  -F "description=100% cotton, round neck"