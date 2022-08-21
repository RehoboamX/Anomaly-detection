from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
from models.WGAN_GP import Generator, Discriminator
from dataload.dataset import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"experiments/anogan_test", help="path to save experiments results")
parser.add_argument("--dataset", default="1", help="mnist")
parser.add_argument('--dataroot', default=r"numclass9", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=20000, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--size", type=int, default=640, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=64, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--gf_dim", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--df_dim", type=int, default=64, help="channels of middle layers for discriminator")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--ano_param", type=float, default=0.2, help="weights of reconstruction error and disc loss")
parser.add_argument("--gen_pth", default=r"experiments/anogan_train/gen_299.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"experiments/anogan_train/disc_299.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)


## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
test_dataset = load_dataset(opt.dataroot, opt.dataset, opt.size, trans=None, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize)

## model
gen = Generator(opt.nz, opt.nc, opt.gf_dim, ksize=4).to(device)
disc = Discriminator(opt.nc, opt.df_dim).to(device)
gen.load_state_dict(torch.load(opt.gen_pth))
disc.load_state_dict(torch.load(opt.disc_pth))
print("Pretrained models have been loaded.")

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1, requires_grad=True, device=device)


opt.dataSize = test_dataset.__len__()


## def
def splitImage(img, size):
    if img.size(3) % size != 0:
        return
    num = int(img.size(3) / size)
    results = torch.zeros(num**2, img.size(1), size, size)
    split1 = torch.split(img, size, dim=2)
    for i in range(num):
        split2 = torch.split(split1[i], size, dim=3)
        for j in range(num):
            results[i*num+j, :, :, :] = split2[j]
    return results


def catImage(imgs, size):
    if imgs.size(0) != size[0] * size[1]:
        return
    results = torch.zeros(1, imgs.size(1), imgs.size(2)*size[0], imgs.size(3)*size[1])
    width = imgs.size(2)
    height = imgs.size(3)
    for i in range(size[0]):
        for j in range(size[1]):
            results[0, :, i*width:(i+1)*width, j*height:(j+1)*height] = imgs[i*size[0]+j]
    return results



## testing
gen.eval()
disc.eval()

writer_real = SummaryWriter(f"logs/test/real")
writer_fake = SummaryWriter(f"logs/test/fake")

loss = []
tqdm_loader = tqdm.tqdm(test_dataloader)
for i, test_input in enumerate(tqdm_loader):
    tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
    patches_num = int(test_input.size(3) / opt.imageSize)

    test_inputs = splitImage(test_input, opt.imageSize).to(device)  # 16 x 1 x 128 x 128
    z_inputs = gen_z_gauss(test_inputs.size(0), opt.nz).to(device)
    ones = torch.ones(test_inputs.size(0), 1).to(device)    # 充当雅各比向量积中的向量
    optimizerZ = optim.Adam([z_inputs], lr=opt.lr, betas=(opt.b1, opt.b2))

    ## inference
    record = 0
    with tqdm.tqdm(range(opt.n_epoches)) as t_epoches:
        for epoch in t_epoches:
            t_epoches.set_description(f"Epoch {epoch+1} /{opt.n_epoches}")
            ##
            optimizerZ.zero_grad()
            ano_G = gen(z_inputs)
            feature_ano_G, _ = disc(ano_G)
            feature_input, _ = disc(test_inputs)

            residual_loss = torch.sum(torch.abs(test_inputs-ano_G), dim=[1, 2, 3])  # 16 x 1
            disc_loss = torch.sum(torch.abs(feature_ano_G-feature_input), dim=[1, 2, 3])

            total_loss = (1.0-opt.ano_param) * residual_loss + (opt.ano_param) * disc_loss
            total_loss = total_loss.view(test_inputs.size(0), 1)    # 16 x 1
            ##
            # writer.add_scalar("total loss", total_loss.)
            total_loss.backward(ones)                   # 这里需要对，每个输入进行优化，注意backward的参数
            optimizerZ.step()
            t_epoches.set_postfix(total_loss=total_loss.detach().cpu().flatten(), grad=z_inputs[0, 0, 0, 0].item())

            if record % opt.sample_interval == 0:
                with torch.no_grad():

                    img_grid_real = utils.make_grid(test_inputs[:test_inputs.size(0)], normalize=True, nrow=patches_num)
                    img_grid_fake = utils.make_grid(ano_G[:test_inputs.size(0)], normalize=True, nrow=patches_num)

                    writer_real.add_image("Real", img_grid_real, global_step=record)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=record)

            record += 1

            if epoch % 1000 == 0:
                # loss.append(total_loss.detach().cpu().flatten())
                ano_G = gen(z_inputs)
                ano_G = catImage(ano_G, (patches_num, patches_num))  # 1 x 1 x 512 x 512
                residule = torch.abs(test_input-ano_G)

                utils.save_image(torch.cat((test_input, ano_G, residule), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i))
    #utils.save_image(mask, '{0}/{1}-1.png'.format(opt.experiment, i))

    residule = residule.detach().cpu().numpy()
    residule = (residule - np.min(residule)) / (np.max(residule) - np.min(residule)) * 255
    residule = residule.astype("uint8")
    residule = np.transpose(residule[0], [1, 2, 0])
    residule = cv2.cvtColor(residule, cv2.COLOR_BGR2GRAY)
    residule = cv2.applyColorMap(residule, cv2.COLORMAP_JET)
    cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule)


