import argparse
import time
import mytorch
from dataset import ImageFolderDataset
from dataloader import DataLoader
from model import SimpleCNN
from utils import load_model, count_parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved weights')
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"Loading test dataset from {args.data_dir}...")
    t0 = time.time()
    dataset = ImageFolderDataset(args.data_dir)
    load_time = time.time() - t0
    print(f"Dataset Loading Time: {load_time:.4f} seconds")

    # 2. Init Model
    num_classes = len(dataset.class_to_idx)
    model = SimpleCNN(num_classes)
    
    # 3. Load Weights
    load_model(model, args.model_path)
    
    # 4. Evaluation Loop
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    t_eval = time.time()
    
    for X, Y in loader:
        out = model.forward(X)
        
        # Get prediction (argmax)
        # out.data is a list of floats. Shape [1, num_classes]
        logits = out.data
        pred = logits.index(max(logits))
        
        if pred == Y[0]:
            correct += 1
        total += 1

    acc = 100.0 * correct / total
    duration = time.time() - t_eval
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Total Time: {duration:.2f}s")
    print("-" * 30)

if __name__ == "__main__":
    main()