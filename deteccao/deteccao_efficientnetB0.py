import torch
from torchvision import transforms, models
from PIL import Image

# Caminho do modelo e da imagem de teste
model_path = "deteccao/efficientnet_macaco.pth"

img_paths = [
    "deteccao/imagens_teste/capivara.jpg",
    "deteccao/imagens_teste/macaco_prego.jpg",
    "deteccao/imagens_teste/orangotango.jpg",
    "deteccao/imagens_teste/macaco_prego2.jpg",
    "deteccao/imagens_teste/tamandua.jpg"
             ]

#img_path = "deteccao/imagens_teste/tamandua.jpg"

# Transformação igual ao treino com EfficientNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Carregar modelo EfficientNet-B0
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

classes = ["macaco-prego", "outros"]
print("-----------EfficientNetB0--------------")

for path in img_paths:
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0)  # adiciona batch
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    print(f"A imagem é classificada como: {classes[pred.item()]}")
