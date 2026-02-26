from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("full_driver_drowsiness_model.pth", map_location=device)
model.eval()

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