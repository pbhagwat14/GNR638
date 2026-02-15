import json

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        size = 1
        for dim in p.shape:
            size *= dim
        total_params += size
    return total_params

def calculate_flops(model, input_shape=(3,32,32)):
    total_macs = 0
    
    # Conv1: 3x3x3 -> 16 filters. Output 32x32.
    total_macs += (27) * (32 * 32 * 16)
    
    # Conv2: 3x3x16 -> 32 filters. Output 16x16.
    total_macs += (144) * (16 * 16 * 32)
    
    # FC1: 2048 -> 128
    total_macs += (2048 * 128)
    
    # FC2: 128 -> 100
    total_macs += (128 * 100)
    
    return total_macs, total_macs * 2


def save_model(model, path):
    weights = model.parameters()
    with open(path, 'w') as f:
        for param in weights:
            # Save as comma-separated values to avoid binary issues
            f.write(",".join([str(x) for x in param.data]) + "\n")
    print(f"Model saved to {path}")

def load_model(model, path):
    weights = model.parameters()
    with open(path, 'r') as f:
        lines = f.readlines()
    if len(lines) != len(weights):
        print("Error: Model structure mismatch")
        return
    for i, line in enumerate(lines):
        weights[i].data = [float(x) for x in line.strip().split(',')]
    print(f"Model loaded from {path}")