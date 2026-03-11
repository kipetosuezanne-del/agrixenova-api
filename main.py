from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import gdown

app = FastAPI(title="AgriXenova AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model and class names from Google Drive
os.makedirs("model", exist_ok=True)

if not os.path.exists("model/best_agrixenova_model_v4.keras"):
    print("📥 Downloading model from Google Drive...")
    gdown.download("https://drive.google.com/uc?id=1UFa-Ef2PuOkhZUvuOJox7hGCa3-KyKaw", "model/best_agrixenova_model_v4.keras", quiet=False)

if not os.path.exists("model/class_names_v4.json"):
    print("📥 Downloading class names...")
    gdown.download("https://drive.google.com/uc?id=1L35pmqL_KzgvLfg8g-P_LJvXahRIdCx_", "model/class_names_v4.json", quiet=False)

print("🌱 Loading AgriXenova model...")
model = tf.keras.models.load_model("model/best_agrixenova_model_v4.keras")
with open("model/class_names_v4.json", "r") as f:
    class_names = json.load(f)
print(f"✅ Model loaded! {len(class_names)} diseases ready!")

