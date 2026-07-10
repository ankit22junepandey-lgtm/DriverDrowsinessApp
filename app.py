from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# MODEL
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*26*26,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.conv_layers(x)
        x=self.fc_layers(x)
        return x


# -----------------------------
# LOAD MODEL
# -----------------------------
model = CNN()

model.load_state_dict(
    torch.load(
        "driver_Adrowsiness_model_fp16.pth",
        map_location=device
    )
)

# Convert FP16 weights back to float32
model.float()

model.eval()


# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    image = Image.open(file).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        score = output.item()

    if score > 0.5:
        result = f"Non Drowsy ({score*100:.2f}%)"
    else:
        result = f"Drowsy ({(1-score)*100:.2f}%)"

    return result


# -----------------------------
# RUN
# -----------------------------
if __name__=="__main__":
    app.run(host="0.0.0.0",port=7860)
