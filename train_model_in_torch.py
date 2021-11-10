import os

os.environ["http_proxy"] = "http://10.10.64.81:7890"
os.environ["https_proxy"] = "http://10.10.64.81:7890"

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch_model import Model

batch_size = 64
MNIST_path = './'
n_epochs = 5

# gpu 加速 模型上传 cuda 数据上传  gpu上保存的模型能够使用cpu load吗？

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data set
train_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                          train=False,
                                          transform=transforms.ToTensor()
                                          )

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

torchmodel = Model(in_channels=784, out_channels=10)
torchmodel.to(device)
# 如果想要查看模型特定的参数 1 for name in model.state_dict(): 2 model.named_parameters() 3 for layer in model.modules():
# for layers in torchmodel.modules():
#     print(layers.requires_grad_()) 固定某层
# 一个是设置不要更新参数的网络层为false，另一个就是在定义优化器时只传入要更新的参数。当然最优的做法是，优化器中只传入requires_grad=True的参数，这样占用的内存会更小一点，效率也会更高。

losses = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=torchmodel.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0, amsgrad=False)

for epoch in range(n_epochs):

    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)

    for i, data in enumerate(tqdm(train_loader)):
        torchmodel.train()
        X_train, y_train = data
        X_train, y_train = Variable(X_train.cuda()), Variable(y_train.cuda())
        outputs = torchmodel(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = losses(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
        if i % 50 == 1:
            print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, ".format(running_loss / len(train_dataset),
                                                                100 * running_correct / len(train_dataset)))

            torch.cuda.empty_cache()

    print(torchmodel.evaluate(torchmodel, test_loader, gpu_config=True, test_loss=None))

torch.save(torchmodel, './models/mnist_Linear_model.pkl')

