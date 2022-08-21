from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
from models.DCGAN import Generator, Discriminator, initialize_weights
from dataload.dataset import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"experiments/anogan_train", help="path to save experiments results")
parser.add_argument("--dataset", default="0", help="mnist")
parser.add_argument('--dataroot', default="numclass9", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=200, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=128, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--gf_dim", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--df_dim", type=int, default=64, help="channels of middle layers for discriminator")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--gen_pth", default=r"experiments/anogan_train/gen.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"experiments/anogan_train/disc.pth", help="pretrained model of disc")
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
train_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=None, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)

## model
gen = Generator(opt.nz, opt.nc, opt.gf_dim, ksize=4).to(device)
disc = Discriminator(opt.nc, opt.df_dim).to(device)
initialize_weights(gen)
initialize_weights(disc)

#if opt.gen_pth:
#    gen.load_state_dict(torch.load(opt.gen_pth))
#    disc.load_state_dict(torch.load(opt.disc_pth))
#    print("Pretrained models have been loaded.")

## adversarial loss
gen_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
disc_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
gen_criteria = nn.BCELoss()
disc_criteria = nn.BCELoss()

## record results
writer_real = SummaryWriter(f"logs/train/real")
writer_fake = SummaryWriter(f"logs/train/fake")

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1).to(device)

fixed_noise = gen_z_gauss(32, opt.nz)
opt.dataSize = train_dataset.__len__()

gen.train()  # 使用 BatchNorm 或 Dropout 时最好加上
disc.train()

## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t_epoches:
    for epoch in t_epoches:
        t_epoches.set_description(f"Epoch {epoch+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        for idx, inputs in enumerate(train_dataloader):
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0

            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            # inputs = inputs.view([batch_size, -1, opt.imageSize, opt.imageSize])
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            # Update "D": max log(D(x)) + log(1-D(G(z))
            disc_optimizer.zero_grad()

            _, D_real = disc(inputs)
            disc_loss_real = disc_criteria(D_real, label_real)
            #disc_loss_real.backward()

            noise = gen_z_gauss(batch_size, opt.nz)
            outputs = gen(noise)
            _, D_fake = disc(outputs.detach())
            disc_loss_fake = disc_criteria(D_fake, label_fake)
            #disc_loss_fake.backward()

            disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
            disc_loss.backward()
            disc_optimizer.step()
            disc_epoch_loss += disc_loss.item() * batch_size

            # Update 'G' : max log(D(G(z)))
            gen_optimizer.zero_grad()
            noise = gen_z_gauss(batch_size, opt.nz)
            outputs = gen(noise)
            _, D_fake = disc(outputs)
            gen_loss = gen_criteria(D_fake, label_real)
            gen_loss.backward()
            gen_optimizer.step()
            gen_epoch_loss += gen_loss.item() * batch_size

            # 保存最优的模型
            if epoch == idx == 0:
                min_loss = gen_epoch_loss
            if gen_epoch_loss < min_loss:
                min_loss = gen_epoch_loss
                torch.save(gen.state_dict(), '{0}/gen_{1}.pth'.format(opt.experiment, epoch))
                torch.save(disc.state_dict(), '{0}/disc_{1}.pth'.format(opt.experiment, epoch))
                print("-----------------Save optimal model-----------------")

            ## print loss
            if record % opt.sample_interval == 0:
                gen.eval()
                disc.eval()
                print(
                    f"Epoch [{epoch + 1}/{opt.n_epoches}] Batch {idx + 1}/{len(train_dataloader)} \
                                  Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}"
                )
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, opt.nc, opt.imageSize, opt.imageSize)  # 其实这里不reshape形状也一样

                    img_grid_real = utils.make_grid(inputs[:32], normalize=True)
                    img_grid_fake = utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=record)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=record)

            record += 1
            gen.train()
            disc.train()


        ## End of epoch
        gen_epoch_loss /= opt.dataSize
        disc_epoch_loss /= opt.dataSize
        t_epoches.set_postfix(gen_epoch_loss=gen_epoch_loss, disc_epoch_loss=disc_epoch_loss)


        if (epoch+1) % 100 == 0:
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_{1}.pth'.format(opt.experiment, epoch))
            torch.save(disc.state_dict(), '{0}/disc_{1}.pth'.format(opt.experiment, epoch))
