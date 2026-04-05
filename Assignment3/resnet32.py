import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split




CONFIG = {
    
    "n"              : 5,        #n=5 → 6*5+2 = 32 layers
    "num_classes"    : 10,
    "stage_filters"  : [16, 32, 64],   # channels per stage
    "init_filters"   : 16,             

    
    "batch_size"     : 128,
    "weight_decay"   : 1e-4,
    "momentum"       : 0.9,
    "base_lr"        : 0.1,
    "total_epochs"   : 182,            
    "lr_milestones"  : [91, 136],      
    "lr_gamma"       : 0.1,

    
    "data_root"      : "./data",
    "num_workers"    : 2,
    "seed"           : 42,
    "save_path"      : "resnet32_best.pth",
    
    "cifar_mean"     : (0.4914, 0.4822, 0.4465),
    "cifar_std"      : (0.2470, 0.2435, 0.2616),
}


def conv3x3(in_channels, out_channels, stride=1):

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def bn_layer(num_features):
    
    return nn.BatchNorm2d(num_features)


def zero_pad_shortcut(x, out_channels, stride):
   
    x_down = x[:, :, ::stride, ::stride]  # (B, C_in, H', W')

   
    pad_size = out_channels - x_down.size(1)
    left_pad  = pad_size // 2
    right_pad = pad_size - left_pad
    zeros_left  = torch.zeros(
        x_down.size(0), left_pad,
        x_down.size(2), x_down.size(3),
        device=x_down.device, dtype=x_down.dtype
    )
    zeros_right = torch.zeros(
        x_down.size(0), right_pad,
        x_down.size(2), x_down.size(3),
        device=x_down.device, dtype=x_down.dtype
    )
    return torch.cat([zeros_left, x_down, zeros_right], dim=1)


class BasicBlock(nn.Module):
   

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First weighted layer
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1   = bn_layer(out_channels)

        # Second weighted layer
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2   = bn_layer(out_channels)

        self.activation = nn.ReLU(inplace=True)

        self._needs_projection = (stride != 1) or (in_channels != out_channels)
        self._stride           = stride
        self._out_ch           = out_channels

    def forward(self, x):
        identity = x  

        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

       
        if self._needs_projection:
            identity = zero_pad_shortcut(x, self._out_ch, self._stride)

        
        out = self.activation(out + identity)
        return out


def _build_stage(in_channels, out_channels, num_blocks, first_stride):
 
    layers = []
    layers.append(BasicBlock(in_channels, out_channels, stride=first_stride))
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNet32(nn.Module):


    def __init__(self, cfg=CONFIG):
        super(ResNet32, self).__init__()
        n      = cfg["n"]                 # 5 for ResNet-32
        f      = cfg["stage_filters"]     # [16, 32, 64]
        f_init = cfg["init_filters"]      # 16

        # ── Layer 1: single 3×3 conv ─────────────────────────
        self.stem = nn.Sequential(
            conv3x3(3, f_init, stride=1),
            bn_layer(f_init),
            nn.ReLU(inplace=True),
        )

        # ── Stages 1–3 ───────────────────────────────────────
        # Stage 1: 32×32 feature maps, 16 channels, n blocks
        self.layer1 = _build_stage(f_init, f[0], num_blocks=n, first_stride=1)

        # Stage 2: 16×16 feature maps, 32 channels, n blocks
        # stride=2 at first block halves H and W, doubles channels
        self.layer2 = _build_stage(f[0], f[1], num_blocks=n, first_stride=2)

        # Stage 3: 8×8 feature maps, 64 channels, n blocks
        self.layer3 = _build_stage(f[1], f[2], num_blocks=n, first_stride=2)

        # ── Head ─────────────────────────────────────────────
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f[2], cfg["num_classes"])

        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)         # (B,  3, 32, 32) → (B, 16, 32, 32)
        x = self.layer1(x)       # (B, 16, 32, 32) → (B, 16, 32, 32)
        x = self.layer2(x)       # (B, 16, 32, 32) → (B, 32, 16, 16)
        x = self.layer3(x)       # (B, 32, 16, 16) → (B, 64,  8,  8)
        x = self.avgpool(x)      # (B, 64,  1,  1)
        x = torch.flatten(x, 1)  # (B, 64)
        x = self.classifier(x)   # (B, 10)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def get_data_loaders(cfg=CONFIG):

    mean, std = cfg["cifar_mean"], cfg["cifar_std"]

    train_tfm = transforms.Compose([
        transforms.Pad(padding=4, fill=0),     
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_set = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=True,
        download=True, transform=train_tfm
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=False,
        download=True, transform=test_tfm
    )

    gen = torch.Generator().manual_seed(cfg["seed"])
    train_set, val_set = random_split(full_train_set, [45000, 5000],
                                      generator=gen)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=256,
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256,
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def top1_accuracy(model, loader, device):
    """Compute top-1 accuracy (%) over all batches in loader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += targets.size(0)
    return 100.0 * total_correct / total_samples


def run_training(cfg=CONFIG):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResNet-32]  device = {device}")

    model = ResNet32(cfg).to(device)
    print(f"[ResNet-32]  Trainable parameters : {model.count_parameters():,}"
          f"  (paper reports ~0.46 M)")

    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["base_lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=False,   
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["lr_milestones"],
        gamma=cfg["lr_gamma"],
    )


    best_val = 0.0
    for epoch in range(1, cfg["total_epochs"] + 1):
        # -- train --
        model.train()
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        # -- log every 10 epochs --
        if epoch % 10 == 0 or epoch == cfg["total_epochs"]:
            val_acc  = top1_accuracy(model, val_loader,  device)
            test_acc = top1_accuracy(model, test_loader, device)
            cur_lr   = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{cfg['total_epochs']}  "
                  f"loss={epoch_loss:.4f}  "
                  f"val_acc={val_acc:.2f}%  "
                  f"test_acc={test_acc:.2f}%  "
                  f"lr={cur_lr:.5f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), cfg["save_path"])

    model.load_state_dict(torch.load(cfg["save_path"]))
    final = top1_accuracy(model, test_loader, device)
    print(f"\n[ResNet-32]  Best checkpoint test accuracy : {final:.2f}%")
    print(f"             Paper reports                  : ~92.49%  "
          f"(7.51% error, Table 6)")


if __name__ == "__main__":
    run_training(CONFIG)