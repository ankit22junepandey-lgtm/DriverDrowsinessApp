import os
import torch
import torch.nn as nn
import gdown
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# =========================
# Model Architecture
# =========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*26*26, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,1)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# =========================
# Download Model from Google Drive
# =========================
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1ebH7LuOb3H8zvbptlpuPNQTNxBhTBkkO"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# Load Model
# =========================
device = torch.device("cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# Image Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
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
# Run App
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
