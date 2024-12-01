import torch
import torch.nn as nn
import torchvision.models.detection
import numpy as np

# Load the model from the .pt file
model_path = "ssd300_vgg16.pt"  # Your local model file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
print("Loading model...")
model = torch.load(model_path, map_location=device)

# Print the model architecture to analyze its structure
print("Model architecture:")
print(model)

# Create a dummy input tensor (e.g., a random image of the size used in the model)
dummy_input = torch.randn(1, 3, 300, 300).to(device)

# Perform a forward pass with the dummy input
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(dummy_input)

# Print the output structure to inspect predictions
print("Output of the model on dummy input:")
print(output)

# Analyze the structure of the output (it should contain boxes, labels, and scores)
if isinstance(output, list):  # Check if the output is a list (usually for models like SSD)
    print(f"Detected {len(output)} elements in the output list.")
    for i, item in enumerate(output):
        print(f"Output {i} contains:")
        for key in item.keys():
            print(f"  {key}: {item[key].shape}")
else:
    print("Model output is not in list format, analyzing the keys directly:")
    for key in output.keys():
        print(f"  {key}: {output[key].shape}")
