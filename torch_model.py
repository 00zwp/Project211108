import torch
from torch.autograd import Variable


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, losses=None, optimizer=None):
        super(Model, self).__init__()
        # 利用torch.nn.Sequential()快速搭建网络模块
        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_channels, out_features=1024),
            torch.nn.Linear(in_features=1024, out_features=512),
            torch.nn.Linear(in_features=512, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=out_channels),
            torch.nn.Softmax())
        # 上面为定义模型结构
        # 下面为模型训练搭建各种参数，同样也可以在外面设置
        self.losses = losses
        self.optimizer = optimizer

    def forward(self, x):
        x = self.dense(x)
        return x

    def evaluate(self, test_model, test_data_loader, gpu_config=False, test_loss=None):
        test_model.eval()
        correct = 0
        avg_loss = 0
        sum_exampls = 0
        for i, data in enumerate(test_data_loader):
            test_data, test_label = data
            if gpu_config:
                test_data, test_label = Variable(test_data.cuda()), Variable(test_label.cuda())
            else:
                test_data, test_label = Variable(test_data), Variable(test_label)
            outputs = test_model(test_data)
            _, pred = torch.max(outputs.data, 1)
            if test_loss is not None:
                avg_loss += test_loss(outputs, test_label).data.item()

            if gpu_config:
                correct += torch.sum(pred == test_label.data).cpu().numpy()
            else:
                correct += torch.sum(pred == test_label.data).numpy()
            sum_exampls += len(test_data)

        torch.cuda.empty_cache()
        if test_loss is not None:
            return correct/sum_exampls, avg_loss/sum_exampls
        return correct/sum_exampls
