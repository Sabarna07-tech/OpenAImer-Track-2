# baseline.py

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import time
import os

def load_baseline_model():
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

def test_with_dummy_input(model):
    # Create a dummy input (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)

def measure_latency(model, iterations=100):
    dummy_input = torch.randn(1, 3, 224, 224)
    start_time = time.time()
    for _ in range(iterations):
        _ = model(dummy_input)
    avg_latency = (time.time() - start_time) / iterations
    print(f"Average latency over {iterations} iterations: {avg_latency:.4f} seconds")
    return avg_latency

def save_model(model, filename='resnet18_baseline.pth'):
    torch.save(model.state_dict(), filename)
    file_size = os.path.getsize(filename) / (1024 * 1024)  # size in MB
    print(f"Model saved: {filename} ({file_size:.2f} MB)")
    return file_size

if __name__ == "__main__":
    model = load_baseline_model()
    test_with_dummy_input(model)
    latency = measure_latency(model)
    size = save_model(model)
