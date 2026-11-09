import os
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

try:
    import streamlit as st
    cache_resource = st.cache_resource
except ImportError:
    def cache_resource(**kwargs):
        def decorator(f):
            return f
        return decorator

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["No Fire", "Fire"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

@cache_resource(show_spinner="Loading Vision Transformer model...")
def load_model(model_path="best_model.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_image(model, img):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=x).logits
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()

    return classes[pred], float(probs[0][pred])
