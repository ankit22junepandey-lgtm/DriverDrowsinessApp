from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

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
            nn.Linear(86528, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = CNNModel()
model.load_state_dict(torch.load("driver_drowsiness_model.pth", map_location="cpu"))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    img = None

    # 1️⃣ Check file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            img = Image.open(file).convert("RGB")

    # 2️⃣ Check webcam base64 image
    elif request.data:
        data = request.data.decode("utf-8")
        image_data = data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if img is None:
        return "No image received"

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.transpose(img,(2,0,1))
    img = torch.tensor(img,dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).int().item()

    return "Drowsy" if pred==1 else "Not Drowsy"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
