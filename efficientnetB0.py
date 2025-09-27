import torch
from torchvision import transforms, models
from PIL import Image

# Caminho do modelo e da imagem de teste
model_path = "efficientnet_macaco.pth"
img_path = "macaco_prego_teste.jpg"  # ou "outros_teste2.jpg"

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
model.load_state_dict(torch.load(model_path))
model.eval()

# Prever imagem
img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0)  # adiciona batch
with torch.no_grad():
    output = model(img_t)
    _, pred = torch.max(output, 1)

classes = ["macaco_prego", "outros"]
print(f"A imagem é classificada como: {classes[pred.item()]}")
