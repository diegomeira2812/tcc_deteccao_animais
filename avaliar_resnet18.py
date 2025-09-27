import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Configurações
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformações (mesmas do treino)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset de validação
val_dataset = datasets.ImageFolder(root="dataset/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modelo
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: macaco_prego e outros
model.load_state_dict(torch.load("resnet18_macaco.pth", map_location=device))
model = model.to(device)
model.eval()

# Avaliação
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Relatório
print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))
