import argparse
import random
import json
import time
import os

import mytorch

from dataset import ImageFolderDataset
from dataloader import DataLoader
from model import SimpleCNN
from utils import count_parameters, calculate_flops, save_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save_path', type=str, default='model_weights.txt', help='Path to save weights')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)

    print("Configuration:", config)

    if 'seed' in config:
        random.seed(config['seed'])
        print(f"Random seed set to {config['seed']}")

    # 2. Load Dataset & Measure Time
    print(f"Loading dataset from {args.data_dir}...")
    t0 = time.time()
    dataset = ImageFolderDataset(args.data_dir)
    load_time = time.time() - t0
    
    print(f"Dataset Loading Time: {load_time:.4f} seconds")
    print(f"Dataset Size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=config['batch_size'])

    # 3. Initialize Model
    # Determine num_classes automatically from dataset
    num_classes = len(dataset.class_to_idx)
    print(f"Detected {num_classes} classes.")
    
    model = SimpleCNN(num_classes)

    # 4. Print Efficiency Metrics
    params = count_parameters(model)
    macs, flops = calculate_flops(model, input_shape=(3,32,32))
    
    print("-" * 30)
    print(f"Model Efficiency Metrics:")
    print(f"Total Parameters: {params}")
    print(f"Total MACs: {macs}")
    print(f"Total FLOPs: {flops}")
    print("-" * 30)

    # 5. Optimizer
    # We need to collect parameters from the model
    # Ensure SimpleCNN has a method to return all params (W and b)
    model_params = model.parameters() 
    opt = mytorch.SGD(model_params, config['learning_rate'])

    # 6. Training Loop
    print("Starting training...")
    total_start = time.time()

    for epoch in range(config['epochs']):
        print(f"\nEPOCH {epoch+1}/{config['epochs']}")
        epoch_loss = 0
        batch_count = 0
        t_epoch = time.time()

        for X, Y in loader:
            # Forward
            out = model.forward(X)
            
            # Loss
            loss = mytorch.cross_entropy(out, Y)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.data[0]
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"  Step {batch_count}: Loss {loss.data[0]:.4f}")

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}. Time: {time.time()-t_epoch:.2f}s")

    print(f"\nTraining completed in {time.time() - total_start:.2f}s")
    
    # 7. Save Model
    save_model(model, args.save_path)

if __name__ == "__main__":
    main()