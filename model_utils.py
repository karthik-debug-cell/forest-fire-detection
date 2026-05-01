import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["No Fire", "Fire"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

def load_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
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