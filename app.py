from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import gdown
import os

app = FastAPI()

# Download model if not already downloaded
MODEL_PATH = "full_driver_drowsiness_model.pth"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=FILE_ID"  # Replace FILE_ID from your link
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = "Non-Drowsy" if output.item() > 0.5 else "Drowsy"
        confidence = output.item() if output.item() > 0.5 else 1 - output.item()

    return {"prediction": prediction, "confidence": confidence}
