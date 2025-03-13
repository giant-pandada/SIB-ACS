import os
import time
import torch
import argparse
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import cv2
import torchvision
from time import time
import utils
import models


parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.25, type=float)
parser.add_argument("--device", default="0")
opt = parser.parse_args()
opt.device = "cuda:" + opt.device

def evaluate():

    print("Start evaluate...")
    config = utils.GetConfig(device=opt.device)
    net = models.SIBACS(config).to(config.device).eval()
    print(os.path.join(config.folder, "model.pth"))
    if os.path.exists(os.path.join(config.folder, "model.pth")):
        if torch.cuda.is_available():
            trained_model = torch.load(os.path.join(config.folder, "model.pth"), map_location=config.device)
        else:
            trained_model = torch.load(os.path.join(config.folder, "model.pth"), map_location="cpu")

        net.load_state_dict(trained_model['net'])
        print("Trained model loaded.")
    else:
        raise FileNotFoundError("Missing trained models.")
    res(config, net, save_img=True)


def res(config, net, save_img):

    tensor2image = torchvision.transforms.ToPILImage()
    net = net.eval()
    file_no = [
        68,
        100,
    ]
    folder_name = [
        "BSD68",
        "Urban100",
    ]
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        mse_total = 0
        time_total = 0
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                x = np.array(image, dtype=np.float32) / 255
                x = transform_image(x)
                x = x[0]
                x = x.float().to(config.device)
                ori_x = x

                x = x.float().to(config.device)

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

                ori = x.to(config.device)
                start = time()

                output = net(ori, sr=opt.rate, phase=9)

                end = time()
                time_total = time_total + end - start
                recon_x = output[:, :, 0:h - h_lack, 0:w - w_lack]

                recon_x = torch.squeeze(recon_x).to("cpu")
                ori_x = torch.squeeze(ori_x).to("cpu")
                recon_x = np.clip(recon_x, 0, 1)

                mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
                mse_total += mse
                p = 10 * np.log10(1 / mse)
                p_total = p_total + p

                ssim = SSIM(recon_x.numpy(), ori_x.numpy(), data_range=1)
                s_total = s_total + ssim

                print("\r=> process {:2} done! Run time for {} is {:5.4f}, PSNR: {:5.2f}, SSIM: {:5.4f}"
                      .format(count_all,file,(end-start), p, ssim))

                if save_img:
                    img_path = "./results/image/{}/".format(int(config.ratio * 100))
                    if not os.path.isdir("./results/image/"):
                        os.mkdir("./results/image/")
                    if not os.path.isdir(img_path):
                        os.mkdir(img_path)
                        print("\rMkdir {}".format(img_path))
                    recon_x = tensor2image(recon_x)
                    recon_x.save(img_path + "({})_{}_{}.png".format(count_all, p, ssim))

            print("=> All the {:2} images done!, your AVG PSNR: {:5.2f}, AVG SSIM: {:5.4f}, AVG TIME: {:5.4f}"
                  .format(file_no[idx], p_total / file_no[idx], s_total / file_no[idx], time_total / file_no[idx]))


if __name__ == "__main__":
    evaluate()
