import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet


def test_data_process():
    test_data = FashionMNIST(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.Resize(size=28),
            transforms.ToTensor()
        ]),
        download=True
    )

    test_dataLoader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )

    return test_dataLoader


def test_model_process(model, test_dataLoader):

    device = "cuda"
    model = model.to(device)

    test_correct = 0
    test_num = 0

    model.eval()

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataLoader:

            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y).item()

            test_num += test_data_x.size(0)

    test_acc = test_correct / test_num

    print("测试准确率:", test_acc)

def answer_test(model,test_dataLoader):
    device = "cuda"
    model = model.to(device)
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataLoader:

            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval() #设置为验证模式

            output= model(test_data_x)
            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab.item()
            label = test_data_y.item()

            print("预测值",result,"真实值",label)



if __name__ == "__main__":

    model = AlexNet()
    model.load_state_dict(torch.load('best_model.pth'))

    test_dataloader = test_data_process()

    test_model_process(model, test_dataloader)
    #answer_test(model, test_dataloader)