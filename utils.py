import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as  np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def create_advfile(dataset, epsilon, original, adv):
    firstpath = "./Adversarial/"+dataset
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    firstpath += "/ori{}_adv{}".format(original, adv)
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    firstpath += "/epsilon{}".format(epsilon)
    if not os.path.exists(firstpath):
        os.mkdir(firstpath)
    return firstpath

class  Mydataset(Dataset):
    def __init__(self, img_path, label_path=None,transform=None,target_transform=None):
        if label_path is None :
            file = open(img_path,'r',encoding='utf-8')
            imgs = []
            for line in file :
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0],int(words[1])))
            self.imgs = imgs
            self.labels = None
            self.transform = transform
            self.target_transfrom=target_transform
        else:
            self.imgs = np.load(img_path).reshape((-1,28,28,1))
            self.labels = np.load(label_path).reshape((-1))
            self.transform = transform
            self.target_transfrom=target_transform

    def __getitem__(self, item):
        if self.labels is None:
            fn, label = self.imgs[item]
            img = Image.open(fn).convert('RGB')
        else:
            img = self.imgs[item]
            label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def adv_train_examples():
    path = "E:/Project211108/Adversarial/FGSM_train/oriall_advall/epsilon0.05/"
    adv_examples = np.load(path + "ClippedAdversarial_examples.npy").reshape((-1, 28, 28, 1))
    true_label = np.load(path + "Clean_label.npy").reshape((-1))
    adv_label = np.load(path + "Adversarial_label.npy").reshape((-1))

    index = np.where(true_label != 5)
    adv = np.copy(adv_examples[index[0]])
    label = np.copy(adv_label[index[0]])
    print(adv.shape)
    for epsilon in range(1, 10, 1):
        path = "E:/Project211108/Adversarial/FGSM_train/oriall_advall/epsilon{}/".format(epsilon / 10)
        adv_examples = np.load(path + "ClippedAdversarial_examples.npy").reshape((-1, 28, 28, 1))
        adv_label = np.load(path + "Adversarial_label.npy").reshape((-1))
        index = np.where(true_label != 5)
        index = np.array(index[0])
        adv = np.concatenate((adv, adv_examples[index]), axis=0)
        label = np.concatenate((label, adv_label[index]), axis=0)
    np.save("./Adversarial/train.npy", adv)
    np.save("./Adversarial/label.npy", label)

def mnist_noniid(dataset, target, Transform):
    imgs = dataset.data
    labels = dataset.targets
    index = np.where(labels!=5)
    imgs = imgs[index]
    labels = labels[index]
    return imgs,labels

def show_confMat(confusion_mat, classes_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_')

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + '.png'))
    plt.close()

def send_data_email_to(title='实验数据',text_send='无', picture_path=None):
    # 从我的qq邮箱发送到谷歌
    mail_host = 'smtp.qq.com'
    mail_user = '308928125@qq.com'
    # 密码(部分邮箱为授权码)
    mail_pass = 'noaktuoegxlqbicj'
    # 邮件发送方邮箱地址
    sender = '308928125@qq.com'
    # 邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    receivers = ['zhuweipeng0728@gmail.com']

    # 设置eamil信息
    # 添加一个MIMEmultipart类，处理正文及附件
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receivers[0]
    message['Subject'] = title

    # 设置email信息
    # 邮件内容设置
    textpart = MIMEText(text_send, 'plain', 'utf-8')

    with open(picture_path, 'rb')as fp:
        picture = MIMEImage(fp.read())
        # 与txt文件设置相似
        picture['Content-Type'] = 'application/octet-stream'
        picture['Content-Disposition'] = 'attachment;filename="1.png"'

    message.attach(textpart)
    message.attach(picture)
    # 登录并发送邮件
    try:
        smtpObj = smtplib.SMTP()
        # 连接到服务器
        smtpObj.connect(mail_host, 25)
        # 登录到服务器
        smtpObj.login(mail_user, mail_pass)
        # 发送
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        # 退出
        smtpObj.quit()
        print('success')
    except smtplib.SMTPException as e:
        print('error', e)  # 打印错误