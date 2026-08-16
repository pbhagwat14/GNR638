import os
import cv2
import time
import mytorch

class ImageFolderDataset:
    def __init__(self, root):
        t0 = time.time()

        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root))

        for label, cls in enumerate(classes):
            self.class_to_idx[cls] = label
            cls_path = os.path.join(root, cls)

            if not os.path.isdir(cls_path):
                continue

            for file in os.listdir(cls_path):
                if file.endswith(".png"):
                    self.samples.append((os.path.join(cls_path, file), label))

        print("Dataset scan time:", time.time() - t0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.resize(img, (32,32))
        img = img.astype("float32") / 255.0

        # HWC → CHW
        img = img.transpose(2,0,1)

        # flatten to list (NO numpy dependency inside framework)
        data = img.reshape(-1).tolist()

        x = mytorch.Tensor(data, [1,3,32,32], False)
        return x, label
