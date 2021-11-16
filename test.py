import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torchvision.transforms import transforms
from tqdm import tqdm
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
imgs = np.load("E:/Project211108/Adversarial/FGSM_train/oriall_advall/epsilon0.1/ClippedAdversarial_examples.npy")
plt.imshow(imgs[1].reshape(28,28))
plt.show()

#

# from torch_model import Model
#
# batch_size = 1
# MNIST_path = './'
#
# model_dict = torch.load("./models/niid_mnist_Linear_model.pkl",map_location=torch.device('cpu'))
# model = Model(in_channels=784, out_channels=10)
# model.load_state_dict(model_dict)
#
# test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
#                                           train=False,
#                                           transform=transforms.ToTensor()
#                                           )
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# array = np.zeros((10,10))
#
# for i, data in enumerate(tqdm(test_loader)):
#     X_train, y_train = data
#     x = y_train.data.numpy()
#
#     outputs = model(Variable(X_train))
#     _, pred = torch.max(outputs.data, 1)
#     y = pred.data.numpy()
#     array[x,y]+=1
#
# from  utils import show_confMat
#
# show_confMat(array, range(0,10), out_dir="./")