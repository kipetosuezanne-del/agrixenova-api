from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI(title="AgriXenova AI API", version="1.0.0")

# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names
print("🌱 Loading AgriXenova model...")
model = tf.keras.models.load_model("model/best_agrixenova_model_v4.keras")
with open("model/class_names_v4.json", "r") as f:
    class_names = json.load(f)
print(f"✅ Model loaded! {len(class_names)} diseases ready!")

# Treatment database
TREATMENTS = {
    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "severity": "Severe",
        "organic": "Remove infected leaves immediately. Spray neem oil mixed with water every 7 days.",
        "chemical": "Apply Mancozeb or Ridomil Gold fungicide every 7-14 days.",
        "prevention": "Avoid overhead watering. Space plants for good airflow. Rotate crops yearly."
    },
    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "severity": "Moderate",
        "organic": "Remove affected leaves. Apply copper-based spray. Mulch around base of plant.",
        "chemical": "Spray Chlorothalonil or Mancozeb fungicide every 7 days.",
        "prevention": "Water at the base only. Avoid wetting leaves. Remove plant debris after harvest."
    },
    "Tomato___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your tomato is healthy! Keep watering at the base and monitor weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
    "Watermelon___anthracnose": {
        "disease": "Watermelon Anthracnose",
        "severity": "Severe",
        "organic": "Remove infected vines. Spray neem oil solution. Ensure good drainage.",
        "chemical": "Apply Mancozeb or Copper Oxychloride fungicide every 10 days.",
        "prevention": "Use disease-resistant seeds. Avoid overhead irrigation. Rotate crops."
    },
    "Watermelon___downy_mildew": {
        "disease": "Watermelon Downy Mildew",
        "severity": "Moderate",
        "organic": "Spray baking soda solution (1 tbsp per liter water). Remove infected leaves.",
        "chemical": "Apply Metalaxyl or Ridomil fungicide immediately.",
        "prevention": "Plant in well-ventilated areas. Water in the morning only."
    },
    "Watermelon___mosaic_virus": {
        "disease": "Watermelon Mosaic Virus",
        "severity": "Severe",
        "organic": "Remove and destroy infected plants immediately. Control aphids with neem oil.",
        "chemical": "No cure — remove infected plants. Use insecticide to kill aphid carriers.",
        "prevention": "Use virus-resistant varieties. Control aphids. Remove weeds around farm."
    },
    "Watermelon___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your watermelon is healthy! Keep up the good work! 🍉",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Maize Common Rust",
        "severity": "Moderate",
        "organic": "Remove heavily infected leaves. Apply neem oil spray every 7 days.",
        "chemical": "Spray Propiconazole or Mancozeb fungicide at first sign of rust.",
        "prevention": "Plant rust-resistant maize varieties. Avoid late planting season."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Maize Northern Leaf Blight",
        "severity": "Severe",
        "organic": "Remove infected leaves. Spray copper-based fungicide.",
        "chemical": "Apply Azoxystrobin or Propiconazole fungicide every 14 days.",
        "prevention": "Use resistant hybrids. Rotate crops. Bury or burn crop debris."
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your maize is healthy! 🌽 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray neem oil or copper fungicide every 7 days.",
        "chemical": "Apply Chlorothalonil or Mancozeb every 7-10 days.",
        "prevention": "Avoid overhead watering. Ensure good plant spacing for airflow."
    },
    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "severity": "Severe",
        "organic": "Remove and destroy all infected plants immediately. Do not compost.",
        "chemical": "Apply Metalaxyl or Ridomil Gold immediately. Repeat every 7 days.",
        "prevention": "Use certified disease-free seed potatoes. Avoid wet conditions."
    },
    "Potato___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your potato is healthy! Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to stay ahead of disease."
    },
    "Mango___Anthracnose": {
        "disease": "Mango Anthracnose",
        "severity": "Severe",
        "organic": "Prune infected branches. Spray neem oil every 7 days during flowering.",
        "chemical": "Apply Mancozeb or Copper Oxychloride before and after flowering.",
        "prevention": "Prune for good airflow. Avoid overhead irrigation. Remove fallen leaves."
    },
    "Mango___Healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your mango tree is healthy! 🥭 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
    "Sugarcane___RedRot": {
        "disease": "Sugarcane Red Rot",
        "severity": "Severe",
        "organic": "Remove and destroy infected stalks immediately. Treat seeds before planting.",
        "chemical": "Soak seed pieces in Carbendazim solution for 30 minutes before planting.",
        "prevention": "Use disease-free planting material. Improve field drainage."
    },
    "Sugarcane___Healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your sugarcane is healthy! Keep monitoring.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
}

def get_treatment(class_name):
    """Get treatment for a disease class"""
    # Direct match
    if class_name in TREATMENTS:
        return TREATMENTS[class_name]
    # Try partial match
    for key in TREATMENTS:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return TREATMENTS[key]
    # Default response
    return {
        "disease": class_name.replace("___", " — ").replace("_", " "),
        "severity": "Moderate",
        "organic": "Remove infected leaves immediately. Apply neem oil spray every 7 days.",
        "chemical": "Consult your local agro-dealer for the right fungicide.",
        "prevention": "Ensure good airflow, avoid overhead watering, rotate crops yearly."
    }

@app.get("/")
def home():
    return {
        "app": "AgriXenova AI",
        "version": "1.0.0",
        "status": "🌱 Running!",
        "diseases": len(class_names),
        "built_by": "Vexanova Tech Hub 🇰🇪"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100
        class_name = class_names[predicted_index]

        # Low confidence warning
        if confidence < 70:
            return {
                "status": "low_confidence",
                "message": "Please retake photo in better lighting",
                "confidence": round(confidence, 2),
                "class": class_name
            }

        # Get treatment
        treatment = get_treatment(class_name)
        is_healthy = "healthy" in class_name.lower()

        return {
            "status": "success",
            "class": class_name,
            "disease": treatment["disease"],
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy,
            "severity": treatment["severity"],
            "treatment": {
                "organic": treatment["organic"],
                "chemical": treatment["chemical"],
            },
            "prevention": treatment["prevention"],
            "built_by": "Vexanova Tech Hub 🇰🇪"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    