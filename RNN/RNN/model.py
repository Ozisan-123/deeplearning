import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)

        for t in range(seq_len):
            xt = x[:, t, :]
            combined = torch.cat([xt, h], dim=1)

            f = torch.sigmoid(self.Wf(combined))
            i = torch.sigmoid(self.Wi(combined))
            c_tilde = torch.tanh(self.Wc(combined))
            o = torch.sigmoid(self.Wo(combined))

            c = f * c + i * c_tilde
            h = o * torch.tanh(c)
        out = self.fc(h)
        return out
        