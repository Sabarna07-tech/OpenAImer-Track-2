# debug_block.py

import torchvision.models as models

def inspect_block():
    # Load the ResNet18 model using the new weights parameter to avoid deprecation warnings.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Access the first block in layer1
    block = model.layer1[0]
    # Print all children (attributes/modules) in the block
    print("Children of model.layer1[0]:")
    for name, module in block.named_children():
        print(f" - {name}: {module}")

    # Alternatively, print the full directory listing
    print("\nComplete dir of block:")
    print(dir(block))

if __name__ == "__main__":
    inspect_block()
