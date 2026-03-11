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
    gdown.download("https://drive.google.com/uc?id=1UFa-Ef2PuOkhZUvuOJox7hGCa3-KyKaw", "model/best_agrixenova_model_v4.keras", quiet=False, fuzzy=True)

if not os.path.exists("model/class_names_v4.json"):
    print("📥 Downloading class names...")
    gdown.download("https://drive.google.com/uc?id=1L35pmqL_KzgvLfg8g-P_LJvXahRIdCx_", "model/class_names_v4.json", quiet=False, fuzzy=True)

print("🌱 Loading AgriXenova model...")
model = tf.keras.models.load_model("model/best_agrixenova_model_v4.keras")
with open("model/class_names_v4.json", "r") as f:
    class_names = json.load(f)
print(f"✅ Model loaded! {len(class_names)} diseases ready!")

# ============================================================
# TREATMENT DATABASE — All 59 diseases
# ============================================================
TREATMENTS = {
    # TOMATO
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
    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray copper-based solution every 7 days.",
        "chemical": "Apply Copper Oxychloride or Streptomycin sulfate spray.",
        "prevention": "Use certified disease-free seeds. Avoid overhead irrigation. Rotate crops."
    },
    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "severity": "Moderate",
        "organic": "Improve air circulation. Remove infected leaves. Spray neem oil weekly.",
        "chemical": "Apply Chlorothalonil or Mancozeb fungicide every 7 days.",
        "prevention": "Space plants properly. Keep greenhouse humidity below 85%. Avoid wetting leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "disease": "Tomato Septoria Leaf Spot",
        "severity": "Moderate",
        "organic": "Remove infected lower leaves. Apply copper fungicide spray every 7-10 days.",
        "chemical": "Spray Mancozeb or Chlorothalonil every 7 days.",
        "prevention": "Mulch around plants. Water at base only. Rotate crops every season."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Tomato Spider Mites",
        "severity": "Mild",
        "organic": "Spray strong water jet to dislodge mites. Apply neem oil every 5 days.",
        "chemical": "Apply Abamectin or Spiromesifen miticide every 7 days.",
        "prevention": "Keep plants well watered. Remove weeds. Introduce natural predators."
    },
    "Tomato___Target_Spot": {
        "disease": "Tomato Target Spot",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray neem oil or copper solution every 7 days.",
        "chemical": "Apply Azoxystrobin or Mancozeb fungicide.",
        "prevention": "Avoid overhead watering. Ensure good plant spacing. Rotate crops."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus",
        "severity": "Severe",
        "organic": "Remove and destroy infected plants. Control whiteflies with neem oil spray.",
        "chemical": "No cure — remove plants. Apply Imidacloprid to control whitefly carriers.",
        "prevention": "Use resistant varieties. Install yellow sticky traps. Control whiteflies early."
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus",
        "severity": "Severe",
        "organic": "Remove and destroy infected plants. Wash hands and tools after handling.",
        "chemical": "No cure available — remove infected plants immediately.",
        "prevention": "Use virus-free seeds. Control aphids. Disinfect tools regularly."
    },
    "Tomato___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your tomato is perfectly healthy! 🍅 Keep watering at the base.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },

    # POTATO
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
        "organic": "Your potato is healthy! 🥔 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to stay ahead of disease."
    },

    # CORN / MAIZE
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
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Maize Gray Leaf Spot",
        "severity": "Moderate",
        "organic": "Remove infected lower leaves. Improve air circulation around plants.",
        "chemical": "Apply Trifloxystrobin or Azoxystrobin fungicide.",
        "prevention": "Plant resistant varieties. Rotate crops. Avoid dense planting."
    },
    "Corn_(maize)___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your maize is healthy! 🌽 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },

    # APPLE
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "severity": "Moderate",
        "organic": "Remove fallen leaves. Spray neem oil every 7-10 days in wet weather.",
        "chemical": "Apply Captan or Mancozeb fungicide from bud break onwards.",
        "prevention": "Plant resistant varieties. Rake and destroy fallen leaves. Prune for airflow."
    },
    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "severity": "Severe",
        "organic": "Prune infected branches. Remove mummified fruit. Apply copper spray.",
        "chemical": "Apply Captan or Thiophanate-methyl fungicide every 10-14 days.",
        "prevention": "Remove dead wood and mummified fruit. Maintain good orchard sanitation."
    },
    "Apple___Cedar_apple_rust": {
        "disease": "Apple Cedar Rust",
        "severity": "Moderate",
        "organic": "Remove nearby cedar trees if possible. Spray neem oil during spring.",
        "chemical": "Apply Myclobutanil or Propiconazole fungicide from pink bud stage.",
        "prevention": "Plant rust-resistant apple varieties. Remove cedar trees within 1km."
    },
    "Apple___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your apple tree is healthy! 🍎 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Prune annually for good airflow. Scan weekly."
    },

    # GRAPE
    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "severity": "Severe",
        "organic": "Remove infected berries and leaves. Apply copper-based spray every 10 days.",
        "chemical": "Apply Mancozeb or Myclobutanil fungicide every 10-14 days.",
        "prevention": "Remove mummified berries. Prune for good air circulation. Remove weeds."
    },
    "Grape___Esca_(Black_Measles)": {
        "disease": "Grape Black Measles",
        "severity": "Severe",
        "organic": "Prune infected wood. Paint wounds with wound sealant. No reliable organic cure.",
        "chemical": "No fully effective chemical — remove severely infected vines.",
        "prevention": "Protect pruning wounds immediately. Use clean tools. Avoid pruning in rain."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Grape Leaf Blight",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray copper solution every 10 days.",
        "chemical": "Apply Mancozeb or Copper Oxychloride fungicide.",
        "prevention": "Ensure good air circulation. Avoid overhead irrigation."
    },
    "Grape___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your grape vine is healthy! 🍇 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Prune annually. Scan weekly for early disease detection."
    },

    # MANGO
    "Mango___Anthracnose": {
        "disease": "Mango Anthracnose",
        "severity": "Severe",
        "organic": "Prune infected branches. Spray neem oil every 7 days during flowering.",
        "chemical": "Apply Mancozeb or Copper Oxychloride before and after flowering.",
        "prevention": "Prune for good airflow. Avoid overhead irrigation. Remove fallen leaves."
    },
    "Mango___Bacterial Canker": {
        "disease": "Mango Bacterial Canker",
        "severity": "Severe",
        "organic": "Prune infected branches. Apply copper-based spray immediately.",
        "chemical": "Spray Copper Oxychloride or Streptomycin sulfate every 15 days.",
        "prevention": "Use disease-free planting material. Disinfect pruning tools. Avoid injuries."
    },
    "Mango___Cutting Weevil": {
        "disease": "Mango Cutting Weevil",
        "severity": "Moderate",
        "organic": "Collect and destroy fallen twigs. Apply neem oil spray on new shoots.",
        "chemical": "Spray Chlorpyrifos or Dimethoate insecticide on new shoots.",
        "prevention": "Remove and destroy infested twigs immediately. Keep orchard clean."
    },
    "Mango___Die Back": {
        "disease": "Mango Die Back",
        "severity": "Severe",
        "organic": "Prune 15cm below infected area. Apply Bordeaux paste on cut ends.",
        "chemical": "Spray Copper Oxychloride or Carbendazim on affected branches.",
        "prevention": "Avoid injuries to bark. Maintain good nutrition. Remove dead wood promptly."
    },
    "Mango___Gall Midge": {
        "disease": "Mango Gall Midge",
        "severity": "Moderate",
        "organic": "Remove and destroy infected flowers and shoots. Apply neem oil spray.",
        "chemical": "Apply Dimethoate or Malathion insecticide at bud burst stage.",
        "prevention": "Collect and destroy fallen infested material. Monitor during flowering."
    },
    "Mango___Healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your mango tree is healthy! 🥭 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
    "Mango___Powdery Mildew": {
        "disease": "Mango Powdery Mildew",
        "severity": "Moderate",
        "organic": "Spray baking soda solution (1 tbsp per liter) every 7 days.",
        "chemical": "Apply Wettable Sulphur or Triadimefon fungicide every 10 days.",
        "prevention": "Ensure good air circulation. Avoid excessive nitrogen fertilizer."
    },
    "Mango___Sooty Mould": {
        "disease": "Mango Sooty Mould",
        "severity": "Mild",
        "organic": "Wash leaves with mild soap solution. Control insects with neem oil.",
        "chemical": "Apply Copper Oxychloride after controlling insect pests.",
        "prevention": "Control aphids and scale insects which cause honeydew. Prune for airflow."
    },

    # SUGARCANE
    "Sugarcane___Healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your sugarcane is healthy! 🌾 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },
    "Sugarcane___Mosaic": {
        "disease": "Sugarcane Mosaic Virus",
        "severity": "Severe",
        "organic": "Remove and destroy infected plants. Control aphids with neem oil spray.",
        "chemical": "No cure — remove infected plants. Control aphid vectors with insecticide.",
        "prevention": "Use disease-free planting material. Control aphids. Remove volunteer plants."
    },
    "Sugarcane___RedRot": {
        "disease": "Sugarcane Red Rot",
        "severity": "Severe",
        "organic": "Remove and destroy infected stalks immediately. Treat seeds before planting.",
        "chemical": "Soak seed pieces in Carbendazim solution for 30 minutes before planting.",
        "prevention": "Use disease-free planting material. Improve field drainage."
    },
    "Sugarcane___Rust": {
        "disease": "Sugarcane Rust",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray neem oil every 7 days.",
        "chemical": "Apply Propiconazole or Mancozeb fungicide every 14 days.",
        "prevention": "Plant resistant varieties. Avoid excessive nitrogen fertilizer."
    },
    "Sugarcane___Yellow": {
        "disease": "Sugarcane Yellow Leaf Disease",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Control aphid vectors with neem oil.",
        "chemical": "Control aphids with Imidacloprid. No direct chemical cure.",
        "prevention": "Use healthy planting material. Control aphid populations early."
    },

    # WATERMELON
    "watermelon___anthracnose": {
        "disease": "Watermelon Anthracnose",
        "severity": "Severe",
        "organic": "Remove infected vines. Spray neem oil solution. Ensure good drainage.",
        "chemical": "Apply Mancozeb or Copper Oxychloride fungicide every 10 days.",
        "prevention": "Use disease-resistant seeds. Avoid overhead irrigation. Rotate crops."
    },
    "watermelon___downy_mildew": {
        "disease": "Watermelon Downy Mildew",
        "severity": "Moderate",
        "organic": "Spray baking soda solution (1 tbsp per liter water). Remove infected leaves.",
        "chemical": "Apply Metalaxyl or Ridomil fungicide immediately.",
        "prevention": "Plant in well-ventilated areas. Water in the morning only."
    },
    "watermelon___mosaic_virus": {
        "disease": "Watermelon Mosaic Virus",
        "severity": "Severe",
        "organic": "Remove and destroy infected plants immediately. Control aphids with neem oil.",
        "chemical": "No cure — remove infected plants. Use insecticide to kill aphid carriers.",
        "prevention": "Use virus-resistant varieties. Control aphids. Remove weeds around farm."
    },
    "watermelon___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your watermelon is perfectly healthy! 🍉 Great job farmer!",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch any disease early."
    },

    # PEPPER
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Pepper Bacterial Spot",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray copper-based solution every 7 days.",
        "chemical": "Apply Copper Hydroxide or Streptomycin sulfate spray.",
        "prevention": "Use certified disease-free seeds. Avoid overhead watering. Rotate crops."
    },
    "Pepper,_bell___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your pepper is healthy! 🌶️ Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },

    # OTHER CROPS
    "Strawberry___Leaf_scorch": {
        "disease": "Strawberry Leaf Scorch",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Apply neem oil spray every 7 days.",
        "chemical": "Apply Captan or Myclobutanil fungicide every 10 days.",
        "prevention": "Avoid overhead watering. Remove old leaves. Rotate strawberry beds."
    },
    "Strawberry___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your strawberry is healthy! 🍓 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Peach___Bacterial_spot": {
        "disease": "Peach Bacterial Spot",
        "severity": "Moderate",
        "organic": "Remove infected leaves. Spray copper-based fungicide every 7 days.",
        "chemical": "Apply Oxytetracycline or Copper Hydroxide spray.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation. Prune for airflow."
    },
    "Peach___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your peach tree is healthy! 🍑 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Cherry Powdery Mildew",
        "severity": "Moderate",
        "organic": "Spray baking soda solution every 7 days. Remove infected leaves.",
        "chemical": "Apply Myclobutanil or Wettable Sulphur fungicide every 10 days.",
        "prevention": "Prune for good airflow. Avoid excessive nitrogen fertilizer."
    },
    "Cherry_(including_sour)___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your cherry tree is healthy! 🍒 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Blueberry___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your blueberry is healthy! 🫐 Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Raspberry___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your raspberry is healthy! Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Soybean___healthy": {
        "disease": "Healthy",
        "severity": "None",
        "organic": "Your soybean is healthy! Keep monitoring weekly.",
        "chemical": "No treatment needed!",
        "prevention": "Scan weekly to catch disease early."
    },
    "Squash___Powdery_mildew": {
        "disease": "Squash Powdery Mildew",
        "severity": "Moderate",
        "organic": "Spray baking soda solution (1 tbsp per liter) every 7 days.",
        "chemical": "Apply Myclobutanil or Triadimefon fungicide every 10 days.",
        "prevention": "Plant resistant varieties. Ensure good spacing. Avoid overhead watering."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Greening Disease",
        "severity": "Severe",
        "organic": "Remove and destroy infected trees. Control psyllid insects with neem oil.",
        "chemical": "No cure — remove infected trees. Apply Imidacloprid to control psyllids.",
        "prevention": "Use certified disease-free trees. Control Asian citrus psyllid aggressively."
    },
}

def get_treatment(class_name):
    if class_name in TREATMENTS:
        return TREATMENTS[class_name]
    for key in TREATMENTS:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return TREATMENTS[key]
    return {
        "disease": class_name.replace("___", " — ").replace("_", " "),
        "severity": "Moderate",
        "organic": "Remove infected leaves immediately. Apply neem oil spray every 7 days.",
        "chemical": "Consult your local agro-dealer for the right fungicide for this disease.",
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
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100
        class_name = class_names[predicted_index]

        top3_indices = np.argsort(predictions[0])[::-1][:3]
        top3 = [
            {
                "disease": class_names[i].replace("___", " — ").replace("_", " "),
                "confidence": round(float(predictions[0][i]) * 100, 2)
            }
            for i in top3_indices
        ]

        if confidence < 70:
            return {
                "status": "low_confidence",
                "message": "Please retake photo in better lighting 📸",
                "confidence": round(confidence, 2),
                "class": class_name,
                "top_3_predictions": top3
            }

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
            "top_3_predictions": top3,
            "built_by": "Vexanova Tech Hub 🇰🇪"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}