import os

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