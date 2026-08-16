import mytorch

class SimpleCNN:
    def __init__(self, num_classes):
        # --- Layer 1 ---
        # 3 input channels -> 16 filters.
        # This is standard for CIFAR-100 starting layers.
        self.conv1 = mytorch.Conv2D(3, 16, 3, 3, 1, 1) 
        self.pool1 = mytorch.MaxPool2D(2) # Output: 16x16x16

        # --- Layer 2 ---
        # 16 input -> 32 filters.
        # We double the depth as we halve the size.
        self.conv2 = mytorch.Conv2D(16, 32, 3, 3, 1, 1)
        self.pool2 = mytorch.MaxPool2D(2) # Output: 8x8x32

        # --- Classifier ---
        self.flat = mytorch.Flatten()
        
        # Flattened size: 32 channels * 8 * 8 = 2048
        # Hidden layer: 128 neurons is plenty for this feature size
        self.fc1 = mytorch.Linear(2048, 128) 
        self.fc2 = mytorch.Linear(128, num_classes)

    def forward(self, x):
        # Standard Conv-ReLU-Pool pattern
        x = self.pool1.forward(mytorch.relu(self.conv1.forward(x)))
        x = self.pool2.forward(mytorch.relu(self.conv2.forward(x)))
        
        x = self.flat.forward(x)
        x = mytorch.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x

    def parameters(self):
        return [
            self.conv1.W, self.conv1.b,
            self.conv2.W, self.conv2.b,
            self.fc1.W, self.fc1.b,
            self.fc2.W, self.fc2.b
        ]