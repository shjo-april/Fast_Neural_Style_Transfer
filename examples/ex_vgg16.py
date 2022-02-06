
import torch
from core import networks

model = networks.VGG16()
model.eval()

images = torch.zeros((1, 3, 448, 448))
features = model(images)

print(images.shape)
print([feature.shape for feature in features])