import os
import random
import shutil
from PIL import Image

# Ajuste para o seu dataset
base_dir = "dataset_raw"   # onde estão suas pastas originais
output_dir = "dataset"     # pasta final organizada
classes = ["macaco_prego", "outros"]
img_size = (224, 224)  # tamanho padrão para ResNet

# Cria estrutura de pastas
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Função para dividir e copiar imagens
def split_dataset(class_name):
    src_dir = os.path.join(base_dir, class_name)
    imgs = os.listdir(src_dir)
    random.shuffle(imgs)

    split_idx = int(0.7 * len(imgs))  # 70% treino
    train_imgs, val_imgs = imgs[:split_idx], imgs[split_idx:]

    for i, img in enumerate(train_imgs):
        try:
            img_path = os.path.join(src_dir, img)
            img_out = os.path.join(output_dir, "train", class_name, f"{class_name}_{i}.jpg")

            im = Image.open(img_path).convert("RGB")
            im = im.resize(img_size)
            im.save(img_out, "JPEG")
        except:
            print(f"Erro ao processar {img}")

    for i, img in enumerate(val_imgs):
        try:
            img_path = os.path.join(src_dir, img)
            img_out = os.path.join(output_dir, "val", class_name, f"{class_name}_{i}.jpg")

            im = Image.open(img_path).convert("RGB")
            im = im.resize(img_size)
            im.save(img_out, "JPEG")
        except:
            print(f"Erro ao processar {img}")

# Roda para cada classe
for cls in classes:
    split_dataset(cls)

print("Dataset organizado em pastas train/ e val/, pronto para treino!")
