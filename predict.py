import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

img_path='test.jpg'
device='cuda' if torch.cuda.is_available() else 'cpu'
tf=transforms.Compose([
transforms.Grayscale(num_output_channels=3),
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize([0.5]*3,[0.5]*3)
])
model=ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',num_labels=2,ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('best_model.pth',map_location=device))
model.to(device).eval()
img=tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
with torch.no_grad():
 out=model(pixel_values=img).logits
 p=out.argmax(1).item()
print('Prediction:', 'Fire' if p==1 else 'No Fire')
