
import torch
from core import networks

model = networks.Transformer(3)
model.eval()

images = torch.zeros((1, 3, 448, 448))
transformed_images = model(images)

print(images.shape)
print(transformed_images.shape)