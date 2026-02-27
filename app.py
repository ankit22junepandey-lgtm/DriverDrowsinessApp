import os
import torch
import gdown
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# =========================
# Download Model From Google Drive
# =========================
MODEL_PATH = "full_driver_drowsiness_model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1ebH7LuOb3H8zvbptlpuPNQTNxBhTBkkO"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# Load FULL Model (FIXED)
# =========================
device = torch.device("cpu")

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# =========================
# Image Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prob = output.item()
            prediction = "Non-Drowsy" if prob > 0.5 else "Drowsy"
            confidence = round(prob, 4)

    return render_template("index.html", prediction=prediction, confidence=confidence)

# =========================
# Render Port Fix
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
