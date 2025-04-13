# quantization.py

import torch
import torchvision.models as models
import torch.quantization as quant

def load_baseline_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def fuse_model(model):
    # Fuse the first few layers; you can fuse more layers as needed.
    # Adjust list of layers depending on model architecture.
    # Here fusing only first conv, bn, and relu.
    model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=False)
    return model_fused

def quantize_model(model):
    model.eval()
    # Set quantization configuration: fbgemm for CPUs
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # Prepare the model for quantization
    torch.quantization.prepare(model, inplace=True)
    # (Calibration) Run a few forward passes with dummy inputs
    dummy_input = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        _ = model(dummy_input)
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    return model

if __name__ == "__main__":
    model = load_baseline_model()
    model_fused = fuse_model(model)
    model_quantized = quantize_model(model_fused)
    torch.save(model_quantized.state_dict(), 'resnet18_quantized.pth')
    print("Quantized model saved as resnet18_quantized.pth")
