import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim, ksize):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gf_dim*8, ksize, 1, 0, bias=False),
            nn.BatchNorm2d(gf_dim*8),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*8, gf_dim*4, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim*4),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*4, gf_dim*2, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*2, gf_dim, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(gf_dim, gf_dim >> 1, ksize, 2, 1, bias=False),
            #nn.BatchNorm2d(gf_dim >> 1),
            #nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim, c_dim, ksize, 2, 1, bias=False),
            nn.Tanh(),
        )


    def forward(self, inputs):
        outputs = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(inputs)))))
        return outputs


class Discriminator(nn.Module):
    def __init__(self, c_dim, df_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(

            nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(df_dim*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(df_dim*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(df_dim*4, df_dim*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(df_dim*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(df_dim*8, df_dim*16, 4, 2, 1, bias=False),
            #nn.InstanceNorm2d(df_dim*16, affine=True),
            #nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(df_dim*8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, inputs):
        h4 = self.layer4(self.layer3(self.layer2(self.layer1(inputs))))
        outputs = self.layer5(h4)
        return h4, outputs.view(-1, 1).squeeze(1)        # by squeeze, get just float not float Tenosor


def initialize_weights(model):
    # Initialize weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)   # DCGAN原文将网络中的权重初始化为0均值0.02标准差的数


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # 每个样本的 CxHxW 对应的 epsilon 为同一值
    interpolated_images = epsilon * real + (1 - epsilon) * fake

    # 计算判别器分数
    _, mixed_scores = critic(interpolated_images)  # BATCH_SIZE x 1 x 1 x 1
    mixed_scores = mixed_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    # 计算判别器分数对插值图像的梯度
    gradient = torch.autograd.grad(  # 这里返回的gradient的shape为 BATCH_SIZE x C x H x W，与inputs形状相同
        inputs=interpolated_images,  # 要被计算导数的叶子节点
        outputs=mixed_scores,  # 待被求导的tensor
        grad_outputs=torch.ones_like(mixed_scores),  # vector-Jacobian 乘积中的 “vector”，outputs不是常数时必须设置此项
        create_graph=True,  # 对反向传播过程中再次构建计算图，可求高阶导数
        retain_graph=True,  # 求导后不释放计算图
    )[0]  # 这里的[0]为梯度的张量本身，[1]则为grad_fn，追踪了上一步的操作函数

    gradient = gradient.view(gradient.shape[0], -1)  # BATCH_SIZE x C*H*W
    gradient_norm = gradient.norm(2, dim=1)  # 对第1维度求L2范数，实际上对每C*H*W个元素求L2范数,形状为BATCH_SIZE
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)  # 形状为BATCH_SIZE
    return gradient_penalty



def print_net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(100, 3, 64, 4).to(device)
    initialize_weights(G)
    D = Discriminator(3, 64).to(device)
    initialize_weights(D)
    summary(G, (100, 1, 1))
    summary(D, (3, 128, 128))


if __name__ == '__main__':
    print_net()