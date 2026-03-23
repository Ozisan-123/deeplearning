from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import VGG16
import torch
import torch.nn as nn
import copy
import time
import pandas as pd

def train_val_data_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ✅ 改成 224×224
        transforms.ToTensor()
    ])

    train_data = FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = Data.random_split(train_data, [train_size, val_size])

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda")

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))

        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corects = 0.0

        train_num = 0
        val_num = 0

        for step, (b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()
            output = model(b_x)

            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output,b_y)

            optimizer.zero_grad()

            loss.backward() #怎么实现的

            optimizer.step() #为什么先反向传播再梯度下降

            train_loss += loss.item() * b_x.size(0)

            train_corrects += torch.sum(pre_lab == b_y)

            train_num += b_x.size(0)

        for step, (b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()

            output = model(b_x)

            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)

            val_corects += torch.sum(pre_lab == b_y)

            val_num += b_x.size(0)

        train_loss_all.append(train_loss/ train_num)
        train_acc_all.append(train_corrects.double().item()/ train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corects.double().item()/val_num)

        print(f"Epoch {epoch}: train loss: {train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch}: val loss: {val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

            time_use = time.time() - since
            print("use time {}s".format(time_use))



    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pth')

    train_process = pd.DataFrame(data = {
                "epoch": range(num_epochs),
                "train_loss_all": train_loss_all,
                "val_loss_all": val_loss_all,
                "train_acc_all": train_acc_all,
                "val_acc_all": val_acc_all,
            }
        
    )
    return train_process
    
if __name__ == "__main__":
    model = VGG16()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20)
