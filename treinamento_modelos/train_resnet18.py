import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# métricas
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, f1_score, roc_auc_score,
                             precision_recall_curve, average_precision_score, roc_curve)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
import os

print("Teste de execução")

# Verificação de CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA disponível! Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA não disponível. Usando CPU.")

print("Iniciando script...")

# Caminho do dataset organizado
data_dir = "../dataset"

# Transformações (normalização padrão ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Carregando datasets...")
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
print(f"Train: {len(train_dataset)} imagens, Val: {len(val_dataset)} imagens")

print("Criando dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
print("Dataloaders prontos!")

print("Inicializando modelo...")
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
print("Modelo carregado e movido para o device!")

# Loss e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Treinamento
epochs = 10
print("Iniciando treino...")

out_dir = os.path.abspath(os.path.join("metrics_outputs", "resnet18"))
os.makedirs(out_dir, exist_ok=True)


def evaluate_and_report(model, loader, device, class_names, prefix="val", out_dir=None):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = softmax(outputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    if len(all_probs) == 0:
        print(f"{prefix}: nenhum batch avaliado.")
        return {}

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cls_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    roc_auc = None
    pr_auc = None
    y_score = None
    try:
        if all_probs.shape[1] == 2:
            y_score = all_probs[:, 1]
            roc_auc = roc_auc_score(all_labels, y_score)
            pr_auc = average_precision_score(all_labels, y_score)
    except Exception:
        roc_auc = None
        pr_auc = None

    # Impressão resumida
    print(f"--- {prefix.upper()} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision macro: {prec_macro:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    if roc_auc is not None:
        print(f"AUC-ROC: {roc_auc:.4f}; AUC-PR: {pr_auc:.4f}")
    else:
        print("AUC-ROC/AUC-PR: não calculado.")

    print("\nClassification report (por classe):")
    print(pd.DataFrame(cls_report).transpose())

    base = os.path.join(out_dir, prefix.lower())

    # Salvar matriz de confusão
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"{prefix} Confusion Matrix")
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    # Salvar ROC e PR se disponíveis
    if y_score is not None:
        fpr, tpr, _ = roc_curve(all_labels, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix} ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_roc_curve.png"), bbox_inches="tight")
        plt.close()

        precision, recall, _ = precision_recall_curve(all_labels, y_score)
        ap = average_precision_score(all_labels, y_score)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix} Precision-Recall Curve")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_pr_curve.png"), bbox_inches="tight")
        plt.close()

    # Salvar CSV com classification_report
    df_report = pd.DataFrame(cls_report).transpose()
    df_report["accuracy_overall"] = acc

    csv_name = os.path.join(out_dir, f"{prefix.lower()}_classification_report.csv")
    df_report.to_csv(csv_name, index=True)
    print(f"{prefix} report salvo em {csv_name}")


    # liberar memória
    del all_probs, all_preds, all_labels, cm, df_report
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc
    }

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

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    print(f"Epoch {epoch+1}/{epochs} finalizada - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Validação e métricas detalhadas
    metrics = evaluate_and_report(model, val_loader, device, val_dataset.classes, prefix=f"val_epoch{epoch+1}", out_dir=out_dir)
    val_acc = metrics.get("accuracy", 0) * 100 if metrics else 0
    print(f"Val Acc (epoch {epoch+1}): {val_acc:.2f}%\n")

# Avaliação final (após todas as epochs)
final_metrics = evaluate_and_report(model, val_loader, device, val_dataset.classes, prefix="val_final")

# Salvar modelo treinado
torch.save(model.state_dict(), os.path.join("deteccao", "resnet18_macaco.pth"))
print("Treinamento concluído e modelo salvo em resnet18_macaco.pth!")
