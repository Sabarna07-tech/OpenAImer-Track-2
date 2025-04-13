# pruning.py

import torch
import torchvision.models as models
import torch.nn.utils.prune as prune

def load_baseline_model():
    # Use the new weights argument to avoid deprecated warnings.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def apply_pruning(model, layer_path, amount=0.2):
    # Access the layer module.
    # For example: layer_path = ('layer1', 0, 'conv2')
    block = getattr(model, layer_path[0])[layer_path[1]]
    # Confirm which attribute to prune (conv2 in this example)
    layer = getattr(block, layer_path[2])
    # Apply L1 unstructured pruning on the layer's weight
    prune.l1_unstructured(layer, name='weight', amount=amount)
    # Remove the pruning reparameterization to make weights permanent
    prune.remove(layer, 'weight')
    return model

if __name__ == "__main__":
    model = load_baseline_model()
    # Use 'conv2' if that is the available attribute as per your debug output.
    model = apply_pruning(model, ('layer1', 0, 'conv2'), amount=0.2)
    torch.save(model.state_dict(), 'resnet18_pruned.pth')
    print("Pruned model saved as resnet18_pruned.pth")
