import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm

DATASET_PATH=r'D:\forest_fire_project\dataset\Data'
TRAIN_DIR=os.path.join(DATASET_PATH,'Train')
TEST_DIR=os.path.join(DATASET_PATH,'Test')
device='cuda' if torch.cuda.is_available() else 'cpu'
transform=transforms.Compose([
transforms.Grayscale(num_output_channels=3),
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize([0.5]*3,[0.5]*3)
])
train_full=datasets.ImageFolder(TRAIN_DIR,transform=transform)
test_data=datasets.ImageFolder(TEST_DIR,transform=transform)
n=len(train_full); a=int(n*0.85)
train_data,val_data=random_split(train_full,[a,n-a])
train_loader=DataLoader(train_data,batch_size=8,shuffle=True)
val_loader=DataLoader(val_data,batch_size=8)
model=ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',num_labels=2,ignore_mismatched_sizes=True).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
loss_fn=torch.nn.CrossEntropyLoss()
best=0
for e in range(3):
 model.train()
 for x,y in tqdm(train_loader):
  x,y=x.to(device),y.to(device)
  out=model(pixel_values=x).logits
  loss=loss_fn(out,y)
  opt.zero_grad(); loss.backward(); opt.step()
 model.eval(); c=t=0
 with torch.no_grad():
  for x,y in val_loader:
   x,y=x.to(device),y.to(device)
   p=model(pixel_values=x).logits.argmax(1)
   c+=(p==y).sum().item(); t+=y.size(0)
 acc=100*c/t
 print('Val Acc',acc)
 if acc>best:
  best=acc; torch.save(model.state_dict(),'best_model.pth')
print('Done')
