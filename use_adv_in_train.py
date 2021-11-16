#利用对抗样本进行原始训练，能否一定程度弥补数据异质特点

#21-11-12 由于对抗样本生成有点困难，指向性不明确，这里1.完全使用对抗样本训练，2.提取对抗样本中的 label = 5 类弥补 5类的。

#这里还需要先测试 对抗样本加上一点的随机噪声会怎么样。
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from torch_model import *
from utils import Mydataset,mnist_noniid
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print("model ----- training")
gpu_flag = True
MNIST_path = './'
batch_size = 32
Transform = transforms.Compose([transforms.ToTensor()])

# Data set
train_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                          train=False,
                                          transform=transforms.ToTensor()
                                          )

train_dataset.data,train_dataset.targets = mnist_noniid(train_dataset,target=5,Transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

advDataset = Mydataset(img_path="./Adversarial/train.npy",label_path="./Adversarial/label.npy",transform=Transform)
advDataset_Loader = DataLoader(advDataset,batch_size=batch_size,shuffle=True)

model = Model(in_channels=784, out_channels=10)
if gpu_flag:
    model.cuda()

losses = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0, amsgrad=False)
scheduler = StepLR(optimizer, step_size=2, gamma = 0.3)
n_epochs = 20

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        X_train, y_train = data
        if gpu_flag:
            X_train, y_train = Variable(X_train.cuda()), Variable(y_train.cuda())
        else:
            X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = losses(outputs, y_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        if gpu_flag:
            running_correct += torch.sum(pred == y_train.data).cpu().numpy()
        else:
            running_correct += torch.sum(pred == y_train.data).numpy()

        if i % 50 == 1:
            print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, ".format(running_loss / ((i+1)* train_loader.batch_size),
                                                                100 * running_correct / ((i+1)* train_loader.batch_size)))

            torch.cuda.empty_cache()

    scheduler.step()
    print(evaluate(model, test_loader, gpu_config=gpu_flag, test_loss=None))

    # if epoch % 2 == 1:
    #     state = train(model.state_dict(), advDataset_Loader, losses, gpu_flag)
    #     model.load_state_dict(state)
    #     print(evaluate(model, test_loader, gpu_config=gpu_flag, test_loss=None))

torch.save(model.state_dict(), './models/niid_mnist_Linear_model.pkl')

