import random
import mytorch

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(idxs)

        for i in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[i:i+self.batch_size]

            xs = []
            ys = []

            for j in batch_idxs:
                x,y = self.dataset[j]
                xs.extend(x.data)   # stack manually
                ys.append(y)

            # batch tensor shape [B,3,32,32]
            B = len(batch_idxs)
            x_batch = mytorch.Tensor(xs, [B,3,32,32], False)

            yield x_batch, ys
