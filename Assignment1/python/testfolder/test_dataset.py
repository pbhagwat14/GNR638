from dataset import ImageFolderDataset
from dataloader import DataLoader

ds = ImageFolderDataset("python/data_2")
loader = DataLoader(ds,batch_size=8)

for X,Y in loader:
    print("batch:",X.shape,len(Y))
    break
