from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet,Residual
import torch
import torch.nn as nn
import copy
import time
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def to_rgb(img):
    return img.convert("RGB")

def train_val_data_process():
    transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

    dataset = datasets.Caltech256(
        root='./data',
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")

        # ================== TRAIN ==================
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()

            output = model(b_x)
            loss = criterion(output, b_y)

            # ❗ 防 NaN
            if torch.isnan(loss):
                print("Loss is NaN, stop training")
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            _, preds = torch.max(output, 1)

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(preds == b_y).item()
            train_num += b_x.size(0)

        # ================== VALID ==================
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():  
            for b_x, b_y in val_dataloader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                loss = criterion(output, b_y)

                _, preds = torch.max(output, 1)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(preds == b_y).item()
                val_num += b_x.size(0)

        # ================== 统计 ==================
        train_loss_epoch = train_loss / train_num
        train_acc_epoch = train_corrects / train_num
        val_loss_epoch = val_loss / val_num
        val_acc_epoch = val_corrects / val_num

        train_loss_all.append(train_loss_epoch)
        train_acc_all.append(train_acc_epoch)
        val_loss_all.append(val_loss_epoch)
        val_acc_all.append(val_acc_epoch)

        print(f"Epoch {epoch}: train loss: {train_loss_epoch:.4f} train acc: {train_acc_epoch:.4f}")
        print(f"Epoch {epoch}: val loss: {val_loss_epoch:.4f} val acc: {val_acc_epoch:.4f}")

        # 保存最佳模型
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        print(f"lr: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"time used: {time.time() - since:.2f}s\n")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pth')

    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "val_loss": val_loss_all,
        "train_acc": train_acc_all,
        "val_acc": val_acc_all,
    })

    return train_process  
if __name__ == "__main__":
    model = ResNet(Residual)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20)
