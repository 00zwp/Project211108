#利用对抗样本进行原始训练，能否一定程度弥补数据异质特点

#21-11-12 由于对抗样本生成有点困难，指向性不明确，这里1.完全使用对抗样本训练，2.提取对抗样本中的 label = 5 类弥补 5类的。

#这里还需要先测试 对抗样本加上一点的随机噪声会怎么样。
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch_model import *
from utils import Mydataset
import numpy as np

print("model ----- loading")
gpu_flag = False
device = torch.device("cuda" if gpu_flag else "cpu")
Mymodel = Model(784,10)
if gpu_flag:
    checkpoint = torch.load('./models/mnist_Linear_model.pkl')
else:
    checkpoint = torch.load('./models/mnist_Linear_model.pkl', map_location=torch.device('cpu'))
Mymodel.load_state_dict(checkpoint)
Mymodel.eval()

print("model ----- evaluate")
MNIST_path = './'
batch_size = 32
Transform = transforms.Compose([transforms.ToTensor()])

path = "E:/Project211108/Adversarial/FGSM_train/oriall_advall/epsilon0.1/"
adv_examples = np.load(path+"ClippedAdversarial_examples.npy").reshape((-1,28,28,1))
true_label = np.load(path+"Clean_label.npy").reshape((-1))
adv_label = np.load(path+"Adversarial_label.npy").reshape((-1))

Dataset = Mydataset(img_path=path+"ClippedAdversarial_examples.npy", label_path=path+"Adversarial_label.npy",transform=Transform)
Dataset_Loader = DataLoader(Dataset, batch_size=16, shuffle=False)
print(evaluate(Mymodel, Dataset_Loader))
