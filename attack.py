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
batch_size = 4096
test_dataset = torchvision.datasets.MNIST(root=MNIST_path,
                                          train=True,
                                          transform=transforms.ToTensor()
                                          )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=4,
                                          shuffle=False, num_workers=2,
                                          pin_memory=True)

bounds = (0, 1)
fmodel = fb.PyTorchModel(Mymodel, bounds=bounds, preprocessing=None,device=device)
# fmodel = fmodel.transform_bounds((0,1))


epsilon_select = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
original_label = "all"
targetlabel = "all"

for epsilon in epsilon_select:

    clean_examples = torch.tensor([], device='cuda:0')
    true_labels = torch.tensor([], device='cuda:0')
    adversarial_examples = torch.tensor([], device='cuda:0')
    clipped_adversarial_examples = torch.tensor([], device='cuda:0')
    adversarial_labels = torch.tensor([], device='cuda:0')
    attack = fb.attacks.FGSM()
    # save_path = "./Adversarial/FGSM/epsilon_{}_ori{}_adv{}_test".format(epsilon, original_label, targetlabel)
    save_path = create_advfile("FGSMAttack_train", epsilon, original_label, targetlabel)
    # criterion = TargetedMisclassification(torch.tensor([targetlabel] * 1, device=device))
    # criterion = Misclassification(torch.tensor([original_label] * 1, device=device))
    for i, (test_data,test_label) in enumerate(tqdm(test_loader)):
        if True: #test_label.data == original_label
            test_data = test_data.cuda()
            test_label = test_label.cuda()

            predict_label = torch.max(Mymodel(Variable(test_data)),1)[1]

            classify_right = torch.where(test_label==predict_label)
            test_label = test_label[classify_right]
            test_data = test_data[classify_right]

            raw, clipped, is_adv = attack(fmodel, test_data, test_label,epsilons=epsilon)
            adversarial_label = torch.max(Mymodel(Variable(clipped)),1)[1]

            locations = torch.where(is_adv==True)

            clean_examples = torch.cat([clean_examples,test_data[locations]],dim=0)
            true_labels = torch.cat([true_labels,test_label[locations].float()],dim=0)
            adversarial_examples= torch.cat([adversarial_examples,raw[locations]], dim=0)
            clipped_adversarial_examples = torch.cat([clipped_adversarial_examples, clipped[locations]], dim=0)
            adversarial_labels = torch.cat([adversarial_labels, adversarial_label[locations].float()], dim=0)

    # np.save("{}/Clean_examples.npy".format(save_path),clean_examples.cpu().numpy())
    # np.save("{}/Adversarial_examples.npy".format(save_path),adversarial_examples.cpu().numpy())
    # np.save("{}/ClippedAdversarial_examples.npy".format(save_path),clipped_adversarial_examples.cpu().numpy())
    # np.save("{}/Clean_label.npy".format(save_path),true_labels.cpu().int().numpy())
    # np.save("{}/Adversarial_label.npy".format(save_path),adversarial_labels.cpu().int().numpy())
    torch.cuda.empty_cache()
    del clean_examples, adversarial_examples,clipped_adversarial_examples,true_labels,adversarial_labels