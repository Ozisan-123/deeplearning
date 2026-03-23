import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU()

        self.c1 = nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1)
        self.s2 = nn.MaxPool2d(2, 2)

        self.c3 = nn.Conv2d(96, 256, kernel_size=3, padding=1)
        self.s4 = nn.MaxPool2d(2, 2)

        self.c5 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.s8 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.f1 = nn.Linear(256 * 3 * 3, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):

        x = self.relu(self.c1(x))
        x = self.s2(x)

        x = self.relu(self.c3(x))
        x = self.s4(x)

        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))

        x = self.s8(x)

        x = self.flatten(x)

        x = self.relu(self.f1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.relu(self.f2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.f3(x)

        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device)

    print(summary(model, input_size=(1, 28, 28)))