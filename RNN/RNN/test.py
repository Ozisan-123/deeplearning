import torch
import torch.utils.data as Data
from model import LSTM

def test_data_process():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv("stock.csv")
    df = df.sort_values(by='Date')

    close_price = df['Close'].values.reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(close_price)

    # 构造序列
    def create_dataset(data, seq_len=20):
        x, y = [], []
        for i in range(len(data) - seq_len):
            x.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        import numpy as np
        x = np.array(x)
        y = np.array(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X, Y = create_dataset(data)

    test_data = Data.TensorDataset(X, Y)

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=32,
        shuffle=False
    )

    return test_loader, scaler

def test_model_process(model, test_loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = torch.nn.MSELoss()

    test_loss = 0.0
    test_num = 0

    model.eval()

    with torch.no_grad():
        for test_x, test_y in test_loader:

            test_x = test_x.to(device)
            test_y = test_y.to(device)

            output = model(test_x).view(-1)

            loss = criterion(output, test_y.view(-1))

            test_loss += loss.item() * test_x.size(0)
            test_num += test_x.size(0)

    print("测试集MSE:", test_loss / test_num)

def answer_test(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for test_x, test_y in test_loader:

            test_x = test_x.to(device)
            test_y = test_y.to(device)

            output = model(test_x).view(-1)

            for i in range(len(output)):
                pred = output[i].item()
                real = test_y[i].item()

                print("预测值:", round(pred, 4), "真实值:", round(real, 4))

            break  # 只看一批就行

from model import LSTM

if __name__ == "__main__":

    model = LSTM(1, 64)
    model.load_state_dict(torch.load('best_lstm.pth'))

    test_loader, scaler = test_data_process()

    #test_model_process(model, test_loader)
    answer_test(model, test_loader)