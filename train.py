import os
import time
import torch
import argparse
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.optim.lr_scheduler as LS
from skimage.metrics import structural_similarity as SSIM
import torchvision
import cv2
import random

import models
import utils

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--bs", default=16, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
opt = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_p(config, net,phase):
    torch.cuda.empty_cache()
    net = net.eval()
    file_no = [
        11,
    ]

    folder_name = [
        "Set11",
    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])

    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        path = "{}/".format(config.test_path) + item
        print(path)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        files = os.listdir(path)
        count_all = 0
        with torch.no_grad():
            for file in files:
                count_all = count_all + 1
                print(path + "/" + file)
                image = cv2.imread(path + "/" + file)
                image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                x = np.array(image1, dtype=np.float32) / 255
                x = transform_image(x)
                x = x[0]
                x = x.float().to(config.device)
                ori_x = x

                h = x.size()[0]
                h_lack = 0
                w = x.size()[1]
                w_lack = 0

                if h % config.phi_size != 0:
                    h_lack = config.phi_size - h % config.phi_size
                    temp_h = torch.zeros(h_lack, w).to(config.device)
                    h = h + h_lack
                    x = torch.cat((x, temp_h), 0)

                if w % config.phi_size != 0:
                    w_lack = config.phi_size - w % config.phi_size
                    temp_w = torch.zeros(h, w_lack).to(config.device)
                    w = w + w_lack
                    x = torch.cat((x, temp_w), 1)

                x = torch.unsqueeze(x, 0)
                x = torch.unsqueeze(x, 0)

                sr = opt.rate
                ori = x.to(config.device)
                output = net(ori, sr,phase)

                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]
                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = ori_x.to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                ssim = SSIM(recon_x.detach().numpy(), ori_x.detach().numpy(), data_range=1)
                p = 10 * np.log10(1 / mse)
                s_total = s_total + ssim
                p_total = p_total + p

            return p_total / file_no[idx], s_total / file_no[idx]

def val_q(config, net,phase):
    torch.cuda.empty_cache()
    net = net.eval()
    file_no = [
        68,
    ]

    folder_name = [
        "BSD68",
    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])

    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        path = "{}/".format(config.test_path) + item
        print(path)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        files = os.listdir(path)
        count_all = 0
        with torch.no_grad():
            for file in files:
                count_all = count_all + 1
                print(path + "/" + file)
                image = cv2.imread(path + "/" + file)
                image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                x = np.array(image1, dtype=np.float32) / 255
                x = transform_image(x)
                x = x[0]

                x = x.float().to(config.device)
                ori_x = x

                h = x.size()[0]
                h_lack = 0
                w = x.size()[1]
                w_lack = 0

                if h % config.phi_size != 0:
                    h_lack = config.phi_size - h % config.phi_size
                    temp_h = torch.zeros(h_lack, w).to(config.device)
                    h = h + h_lack
                    x = torch.cat((x, temp_h), 0)

                if w % config.phi_size != 0:
                    w_lack = config.phi_size - w % config.phi_size
                    temp_w = torch.zeros(h, w_lack).to(config.device)
                    w = w + w_lack
                    x = torch.cat((x, temp_w), 1)

                x = torch.unsqueeze(x, 0)
                x = torch.unsqueeze(x, 0)

                sr = opt.rate
                ori = x.to(config.device)
                output = net(ori, sr,phase)

                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]
                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = ori_x.to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                ssim = SSIM(recon_x.detach().numpy(), ori_x.detach().numpy(), data_range=1)
                p = 10 * np.log10(1 / mse)
                s_total = s_total + ssim
                p_total = p_total + p

            return p_total / file_no[idx], s_total / file_no[idx]
def val_m(config, net,phase):
    torch.cuda.empty_cache()
    net = net.eval()
    file_no = [
        100,
    ]

    folder_name = [
        "Urban100",
    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])

    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        path = "{}/".format(config.test_path) + item
        print(path)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        files = os.listdir(path)
        count_all = 0
        with torch.no_grad():
            for file in files:
                count_all = count_all + 1
                print(path + "/" + file)
                image = cv2.imread(path + "/" + file)
                image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                x = np.array(image1, dtype=np.float32) / 255
                x = transform_image(x)
                x = x[0]

                x = x.float().to(config.device)
                ori_x = x

                h = x.size()[0]
                h_lack = 0
                w = x.size()[1]
                w_lack = 0

                if h % config.phi_size != 0:
                    h_lack = config.phi_size - h % config.phi_size
                    temp_h = torch.zeros(h_lack, w).to(config.device)
                    h = h + h_lack
                    x = torch.cat((x, temp_h), 0)

                if w % config.phi_size != 0:
                    w_lack = config.phi_size - w % config.phi_size
                    temp_w = torch.zeros(h, w_lack).to(config.device)
                    w = w + w_lack
                    x = torch.cat((x, temp_w), 1)

                x = torch.unsqueeze(x, 0)
                x = torch.unsqueeze(x, 0)

                sr = opt.rate
                ori = x.to(config.device)
                output = net(ori,sr,phase)

                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]

                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = ori_x.to("cpu")

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                ssim = SSIM(recon_x.detach().numpy(), ori_x.detach().numpy(), data_range=1)
                p = 10 * np.log10(1 / mse)
                s_total = s_total + ssim
                p_total = p_total + p

            return p_total / file_no[idx], s_total / file_no[idx]
def main():
    device = "cuda:" + opt.device
    config = utils.GetConfig(ratio=opt.rate, device=device)
    config.check()
    set_seed(22)
    print("Data loading...")
    torch.cuda.empty_cache()
    dataset_train = utils.train_loader(batch_size=opt.bs)
    net = models.SIBACS(config).to(config.device)
    best = 0
    optimizer = optim.Adam(net.parameters(), lr=20e-5)
    scheduler = LS.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.0001,
                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    if os.path.exists(config.model):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.model, map_location=config.device)['net'])
        else:
            net.load_state_dict(torch.load(config.model, map_location="cpu")['net'])

        print("Loaded trained model of res: {:8.4f}.".format(best))
    start_epoch = 1
    phase = 1   # (From 1 to 9)
    for name, param in net.named_parameters():
        for i in range(phase-1):
            param0 = 'phi.{}'.format(i)
            param1 = 'conv_start_rec.{}'.format(i)
            param2 = 'DUN_mask.{}'.format(i)
            param3 = 'conv_end.{}'.format(i)
            if param0 in name:
                param.requires_grad = False
            if param1 in name:
                param.requires_grad = False
            if param2 in name:
                param.requires_grad = False
            if param3 in name:
                param.requires_grad = False

    for name, param in net.named_parameters():
        print(name, param.requires_grad)

    net.train()
    over_all_time = time.time()
    L1loss = torch.nn.L1Loss(size_average=True, reduce=True)
    SSIMloss = SSIMLoss()
    for epoch in range(start_epoch, config.epochs):
        print("Lr: {:.5e}.".format(optimizer.param_groups[0]['lr']))
        epoch_loss = 0
        dic = {"rate": config.ratio, "epoch": epoch,
               "device": config.device}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):

            xi = xi.to(config.device)
            optimizer.zero_grad()
            sr = random.randint(10, 50) / 100

            xo = net(xi, sr, phase)

            loss = L1loss(xo, xi) + 0.1 * (1 - SSIMloss(xo, xi))
            batch_loss = loss
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                tqdm.write(
                    "\r[{:5}/{:5}], Loss_mean: [{:8.6f}]"
                    .format(config.batch_size * (idx + 1),
                            dataset_train.__len__() * config.batch_size,
                            batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]"
              .format(epoch, avg_loss))
        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        print("\rNow val..")
        p, ps = val_p(config, net, phase)
        q, qs = val_q(config, net, phase)
        m, ms = val_m(config, net, phase)
        print(
            "Set11, PSNR: {:5.2f}, SSIM: {:5.4f}; BSD68, PSNR: {:5.2f}, SSIM: {:5.4f}; Urban100, PSNR: {:5.2f}, SSIM: {:5.4f},"
            .format(p, ps, q, qs, m, ms))
        if epoch % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config.folder, "net_params_%d.pth" % (epoch)))
            print("*", "  Check point of epoch {:2} saved  ".format(epoch).center(120, "="), "*")
            output_file = open(config.log, 'a')
            output_file.write("Epoch {:2.0f}, Loss of train {:8.10f}, Set11, PSNR: {:5.2f}, SSIM: {:5.4f}; BSD68, PSNR: {:5.2f}, SSIM: {:5.4f}; Urban100, PSNR: {:5.2f}, SSIM: {:5.4f}\n".format(epoch, avg_loss, p, ps, q, qs, m, ms))
            output_file.close()
        scheduler.step(p+q+m)
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))
    print("Train end.")


def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(opt.device) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    main()
