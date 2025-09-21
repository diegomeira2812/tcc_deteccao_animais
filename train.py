import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

print("🔹 Iniciando script...")

# Caminho do dataset organizado
data_dir = "dataset"

# Transformações (normalização padrão ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("🔹 Carregando datasets...")
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
print(f"Train: {len(train_dataset)} imagens, Val: {len(val_dataset)} imagens")

print("🔹 Criando dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print("✅ Dataloaders prontos!")

print("🔹 Inicializando modelo...")
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
print("✅ Modelo carregado!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Usando device: {device}")
model = model.to(device)

# Loss e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Treinamento
epochs = 10
print("🔹 Iniciando treino...")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print por batch (a cada 10 batches)
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    acc = 100 * correct / total
    print(f"📊 Epoch {epoch+1}/{epochs} finalizada - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")

    # Validação
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"✅ Val Acc: {val_acc:.2f}%\n")

# Salvar modelo treinado
torch.save(model.state_dict(), "resnet18_macaco.pth")
print("🎉 Treinamento concluído e modelo salvo em resnet18_macaco.pth!")
