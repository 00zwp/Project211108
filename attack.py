from foolbox import TargetedMisclassification, Misclassification
from tqdm import tqdm
from torch_model import *
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import foolbox as fb
import matplotlib.pyplot as plt
from utils import create_advfile
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#利用foolbox包生成对抗样本 —— 1.读取模型生成foolbox包装模型框架  2.attack的执行

gpu_flag = True
device = torch.device("cuda" if gpu_flag else "cpu")

Mymodel = Model(784,10)
if gpu_flag:
    checkpoint = torch.load('./models/mnist_Linear_model.pkl')
else:
    checkpoint = torch.load('./models/mnist_Linear_model.pkl', map_location=torch.device('cpu'))
Mymodel.load_state_dict(checkpoint)
Mymodel = Mymodel.cuda(device)
Mymodel.eval()

MNIST_path = './'
batch_size = 1
test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                          train=True,
                                          transform=transforms.ToTensor()
                                          )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=2,
                                          pin_memory=True)

bounds = (0, 1)
fmodel = fb.PyTorchModel(Mymodel, bounds=bounds, preprocessing=None,device=device)
# fmodel = fmodel.transform_bounds((0,1))
attack = fb.attacks.L2CarliniWagnerAttack()

clean_examples=[]
true_labels=[]
adversarial_examples=[]
clipped_adversarial_examples=[]
adversarial_labels=[]

epsilon_select = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
original_label = "all"
targetlabel = "all"

for epsilon in epsilon_select:
    # save_path = "./Adversarial/FGSM/epsilon_{}_ori{}_adv{}_test".format(epsilon, original_label, targetlabel)

    save_path = create_advfile("CarliniWagnerL2Attack_train", epsilon, original_label, targetlabel)
    # criterion = TargetedMisclassification(torch.tensor([targetlabel] * 1, device=device))
    # criterion = Misclassification(torch.tensor([original_label] * 1, device=device))
    for i, (test_data,test_label) in enumerate(tqdm(test_loader)):
        if True: #test_label.data == original_label
            predict_label = torch.max(Mymodel(Variable(test_data.cuda())),1)[1]
            if test_label == predict_label.data.cpu():
                raw, clipped, is_adv = attack(fmodel, test_data.to(device), test_label.to(device) ,epsilons=epsilon)
                if is_adv:
                    adversarial_label = torch.max(Mymodel(Variable(clipped.cuda())),1)[1]
                    print("iterations in {}: attack true for original {} adversarial {}".format(i,test_label.data,
                                                                                                adversarial_label.data))
                    clean_examples.append(test_data.data.numpy().reshape(1,28,28))
                    true_labels.append(test_label.data.numpy())
                    adversarial_examples.append(raw.data.cpu().numpy().reshape(1,28,28))
                    clipped_adversarial_examples.append(clipped.data.cpu().numpy().reshape(1,28,28))
                    adversarial_labels.append(adversarial_label.data.cpu().numpy())
    torch.cuda.empty_cache()

    np.save("{}/Clean_examples.npy".format(save_path),clean_examples)
    np.save("{}/Adversarial_examples.npy".format(save_path),adversarial_examples)
    np.save("{}/ClippedAdversarial_examples.npy".format(save_path),clipped_adversarial_examples)
    np.save("{}/Clean_label.npy".format(save_path),true_labels)
    np.save("{}/Adversarial_label.npy".format(save_path),adversarial_labels)