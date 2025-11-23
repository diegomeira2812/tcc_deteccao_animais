import os
import tempfile
import traceback
import gc
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

# cria pasta de outputs
out_dir = os.path.abspath(os.path.join("metrics_outputs", "efficientnetB0"))
os.makedirs(out_dir, exist_ok=True)


def evaluate_and_report(model, loader, device, class_names, prefix="val"):
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

    # salvar matriz de confusão
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"{prefix} Confusion Matrix")
    plt.savefig(f"{base}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # salvar ROC e PR se disponíveis
    if y_score is not None:
        fpr, tpr, _ = roc_curve(all_labels, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix} ROC Curve")
        plt.legend()
        plt.savefig(f"{base}_roc_curve.png", bbox_inches="tight")
        plt.close()

        precision, recall, _ = precision_recall_curve(all_labels, y_score)
        ap = average_precision_score(all_labels, y_score)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{prefix} Precision-Recall Curve")
        plt.legend()
        plt.savefig(f"{base}_pr_curve.png", bbox_inches="tight")
        plt.close()

    # salvar CSV com classification_report
    df_report = pd.DataFrame(cls_report).transpose()
    df_report["accuracy_overall"] = acc
    csv_name = f"{base}_classification_report.csv"
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

def main():
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    data_dir = "../dataset"
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} imagens, Val: {len(val_dataset)} imagens")

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes: macaco_prego / outros
    model = model.to(device)
    print("EfficientNet-B0 carregado!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 10
    print("Iniciando treino...")
    history = []

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

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} finalizada - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")

        # Validação
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * correct / total
        print(f"Val Acc: {val_acc:.2f}%\n")

        # métricas detalhadas e salvar outputs
        metrics = evaluate_and_report(model, val_loader, device, val_dataset.classes, prefix=f"val_epoch{epoch+1}")
        history.append({
            "epoch": epoch+1,
            "train_loss": running_loss/len(train_loader) if len(train_loader)>0 else 0,
            "train_acc": acc,
            "val_acc": metrics.get("accuracy", 0)
        })

    # relatório final
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

    # salvar histórico e modelo
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join("deteccao", "efficientnet_macaco.pth"))
    print(f"Treinamento concluído e modelo salvo em efficientnet_macaco.pth! Outputs em: {out_dir}")
    

if __name__ == "__main__":
    main()
