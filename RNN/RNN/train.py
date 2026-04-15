import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import copy
import time
from model import LSTM

def create_dataset(data, seq_len=20):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    x = np.array(x)
    y = np.array(y)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_val_data_process():
    df = pd.read_csv("stock.csv")

    df = df.sort_values(by='Date')

    close_price = df['Close'].values.reshape(-1, 1)

    close_price = pd.DataFrame(close_price).ffill().values

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(close_price)

    # ===== 后面不变 =====
    seq_len = 20
    X, Y = create_dataset(data, seq_len)

    train_size = int(0.8 * len(X))

    train_X, val_X = X[:train_size], X[train_size:]
    train_Y, val_Y = Y[:train_size], Y[train_size:]

    import torch.utils.data as Data
    train_data = Data.TensorDataset(train_X, train_Y)
    val_data = Data.TensorDataset(val_X, val_Y)

    train_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = Data.DataLoader(val_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, scaler


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_loss_all = []
    val_loss_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")

        train_loss = 0.0
        val_loss = 0.0

        train_num = 0
        val_num = 0

        # ===== train =====
        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            output = model(b_x).squeeze()

            loss = criterion(output, b_y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        # ===== val =====
        for b_x, b_y in val_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            with torch.no_grad():
                output = model(b_x).squeeze()
                loss = criterion(output, b_y.squeeze())

            val_loss += loss.item() * b_x.size(0)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        print(f"Epoch {epoch}: train loss {train_loss_all[-1]:.6f}")
        print(f"Epoch {epoch}: val loss {val_loss_all[-1]:.6f}")

        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

            print(f"use time {time.time()-since:.2f}s")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_lstm.pth')

    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "val_loss": val_loss_all
    })

    return train_process


if __name__ == "__main__":
    model = LSTM(input_size=1, hidden_size=64)

    train_loader, val_loader, scaler = train_val_data_process()

    train_process = train_model_process(
        model,
        train_loader,
        val_loader,
        num_epochs=20
    )