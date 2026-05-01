import torch
from torchvision import datasets, transforms
from torch import nn, optim
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "dataset/Data/Train"
TEST_DIR  = "dataset/Data/Test"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_data   = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=4)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(pixel_values=images).logits
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)

with open("metrics.txt", "w") as f:
    f.write(f"{acc},{prec},{rec}")

torch.save(model.state_dict(), "best_model.pth")