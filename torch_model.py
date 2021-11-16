import torch
from torch.autograd import Variable
from tqdm import tqdm


def evaluate(test_model, test_data_loader, gpu_config=False, test_loss=None):
    test_model.eval()
    correct = 0
    avg_loss = 0
    sum_exampls = 0
    for i, data in enumerate(test_data_loader):
        test_data, test_label = data
        # test_data += torch.randn(size=test_data.size())*0.1
        # test_data = torch.clamp(test_data,0,1)
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
        return correct / sum_exampls, avg_loss / sum_exampls
    return correct / sum_exampls

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


    def classify(self,x):
        outputs = self.forward(x)
        confidence, label = torch.max(outputs, 1)
        return int(label.data.numpy()), float(confidence.data.numpy())

def train(model_dict, dataloader, loss, gpu_flag=False):
    model = Model(in_channels=784, out_channels=10)
    model.load_state_dict(model_dict)
    running_loss = 0.0
    running_correct = 0
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, amsgrad=False)
    if gpu_flag:
        model.cuda()
    model.train()
    for i, data in enumerate(tqdm(dataloader)):
        X_train, y_train = data
        if gpu_flag:
            X_train, y_train = Variable(X_train.cuda()), Variable(y_train.cuda())
        else:
            X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        losses = loss(outputs, y_train)
        losses.backward()
        optimizer.step()

        running_loss += losses.data.item()

        if gpu_flag:
            running_correct += torch.sum(pred == y_train.data).cpu().numpy()
        else:
            running_correct += torch.sum(pred == y_train.data).numpy()

        if i % 50 == 1:
            print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, ".format(running_loss /( (i+1)* dataloader.batch_size),
                                                                100 * running_correct /( (i+1)* dataloader.batch_size)))

            torch.cuda.empty_cache()
    return model.state_dict()