import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import ViTForImageClassification

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Data", "Train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Data", "Test")



transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3, [0.5]*3)
])



train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

print("Class mapping:", train_dataset.class_to_idx)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)



model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.to(device)




optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01
)

criterion = torch.nn.CrossEntropyLoss()



EPOCHS = 3

for epoch in range(EPOCHS):

    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images).logits

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Loss:", total_loss)



model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(pixel_values=images).logits

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)

        all_labels.extend(labels.numpy())



accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(
    all_labels,
    all_preds,
    average='weighted'
)

recall = recall_score(
    all_labels,
    all_preds,
    average='weighted'
)

print("\n✅ Accuracy:", accuracy)
print("✅ Precision:", precision)
print("✅ Recall:", recall)



torch.save(model.state_dict(), "best_model.pth")



with open("metrics.txt", "w") as f:

    f.write(f"{accuracy},{precision},{recall}")

print("\n🔥 Training Complete")