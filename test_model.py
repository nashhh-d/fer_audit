import torch

from model import create_model


model = create_model()

print(model)

print("\nFinal layer:")
print(model.fc)


# Test with a fake batch
x = torch.randn(2, 3, 224, 224)

output = model(x)

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)