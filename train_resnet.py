import os
import torch

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Data", "Train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Data", "Test")

# ---------------- TRANSFORMS ----------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ---------------- DATASETS ----------------

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=8)

# ---------------- MODEL ----------------

model = models.resnet50(pretrained=True)

model.fc = torch.nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# ---------------- TRAINING ----------------

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

# ---------------- EVALUATION ----------------

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

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

print("\n📊 RESNET RESULTS")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)