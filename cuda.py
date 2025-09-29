import torch

print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Total de memória:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("GPU não detectada. Verifique drivers e instalação do PyTorch.")

