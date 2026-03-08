from flask import Flask, render_template, request
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

app = Flask(__name__)

# CNN model (must match training architecture)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Load trained model
model = CNNModel()
model.load_state_dict(torch.load("driver_drowsiness_model.pth", map_location="cpu"))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return "Drowsy" if pred.item() == 1 else "Not Drowsy"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
