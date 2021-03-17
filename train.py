from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import utils
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import GeoImage
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
# from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--gpus', default=[0,1,2,3], action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--output_dir', type=str, default="./output/geoimage", help='directory to save output')
parser.add_argument('--net_G', type=str, default="", help='checkpoint of G')
parser.add_argument('--net_D', type=str, default="", help='checkpoint of D')
opt = parser.parse_args()

print(opt)

writer = SummaryWriter("/DATA2/wxr/tensorboard-log")

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
gpus = opt.gpus
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "/home/hanfang_yang/GIS"
train_set = GeoImage(os.path.join(root_path, "data"), opt.direction, "./dataset/list/"+opt.dataset+"/train.lst")
# test_set = get_test_set(root_path + opt.dataset, opt.direction)
test_set = GeoImage(os.path.join(root_path, "data"), opt.direction, "./dataset/list/"+opt.dataset+"/val.lst")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

# device = torch.device("cuda:0" if opt.cuda else "cpu")
# device = torch.cuda.get_device_name(range(opt.gpus))


print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic')

if opt.gpus[-1] > 0:
    net_g = nn.DataParallel(net_g, device_ids=gpus).cuda()
    net_d = nn.DataParallel(net_d, device_ids=gpus).cuda()

if opt.net_G:
    G_checkpoint = torch.load(opt.net_G).state_dict()
    net_g.load_state_dict(G_checkpoint)
    print("=> loaded checkpoint net_G")
if opt.net_D:
    D_checkpoint = torch.load(opt.net_D).state_dict()
    net_d.load_state_dict(D_checkpoint)
    print("=> loaded checkpoint net_D")

criterionGAN = GANLoss().cuda()
criterionL1 = nn.L1Loss().cuda()
criterionB = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b, mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        fake_b = net_g(real_a)
        grid = torchvision.utils.make_grid(fake_b)
        writer.add_image('Pre-damage/Inspect fake data', grid, global_step=0)
        grid = torchvision.utils.make_grid(real_b)
        writer.add_image('Pre-damage/Inspect real data', grid, global_step=0)
        grid = torchvision.utils.make_grid(real_a)
        writer.add_image('Post-damage/Inspect conditional data', grid, global_step=0)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_building_l1 = criterionB(fake_b * mask, real_b * mask) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1 + loss_building_l1
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        writer.add_scalar('Loss/Loss D', loss_d.item(), iteration + epoch * len(training_data_loader))
        writer.add_scalar('Loss/Loss G', loss_g.item(), iteration + epoch * len(training_data_loader))

    for iter in range(fake_b.shape[0]):
        utils.save_img(fake_b[iter].cpu().detach(), batch[2][iter].split("/")[-1], opt.output_dir)

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0

    for batch in testing_data_loader:
        input, target = batch[0].cuda(), batch[1].cuda()

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

writer.close()


