import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ----------------------------
# Define CNN model architecture
# ----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*54*54, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ----------------------------
# Initialize model and load weights
# ----------------------------
device = torch.device('cpu')
model = CNNModel()
model_path = os.path.join(os.getcwd(), "driver_drowsiness_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Make sure it matches your training size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        result = "Drowsy" if predicted.item() == 1 else "Not Drowsy"

    return render_template("index.html", prediction=result)

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
