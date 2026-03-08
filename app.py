from flask import Flask, render_template, request
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

app = Flask(__name__)

# CNN model
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

# Load model
model = CNNModel()
model.load_state_dict(torch.load("driver_drowsiness_model.pth", map_location='cpu'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

# Add route to predict drowsiness from uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((224,224))  # adjust to model input size
    img = np.array(img).transpose((2,0,1))/255.0
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0)
    output = model(img)
    _, pred = torch.max(output, 1)
    return "Drowsy" if pred.item()==1 else "Not Drowsy"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
