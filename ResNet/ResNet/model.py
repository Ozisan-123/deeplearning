import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Residual(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, use_lconv = False,stride=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ReLU(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    def __init__(self,Residual):
        super(ResNet,self).__init__()
        self.bn1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        )
        self.bn2 = nn.Sequential(
            Residual(64,64,use_lconv=False,stride=1),
            Residual(64,64,use_lconv=False,stride=1)
        )
        self.bn3 = nn.Sequential(
            Residual(64,128,use_lconv=True,stride=2),
            Residual(128,128,use_lconv=False,stride=1)
        )  
        self.bn4 = nn.Sequential(
            Residual(128,256,use_lconv=True,stride=2),
            Residual(256,256,use_lconv=False,stride=1)
        )
        self.bn5 = nn.Sequential(
            Residual(256,512,use_lconv=True,stride=2),
            Residual(512,512,use_lconv=False,stride=1)
        )
        self.bn6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,257)
        )

    def forward(self, x):
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)
        x = self.bn4(x)
        x = self.bn5(x)
        x = self.bn6(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(Residual).to(device)
    print(summary(model, input_size=(3, 224, 224)))
    