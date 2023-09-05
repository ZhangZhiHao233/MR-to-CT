import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import time
import logging
import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """
    platform agnostic seed
    :return:
    """
    # note that this still won't be entirely deterministic
    # a better solution can be found at https://github.com/NVIDIA/framework-determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seed_torch(seed: int):
    """ 100% deterministically """
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(log_path='log_path'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path+'/'+timer+'_log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger



class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.register_buffer("kernel", self._cal_gaussian_kernel(11, 1.5))
        self.L = 2.0
        self.k1 = 0.01
        self.k2 = 0.03

    @staticmethod
    def _cal_gaussian_kernel(size, sigma):
        g = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        g = g / g.sum()
        window = g.reshape([-1, 1]).matmul(g.reshape([1, -1]))
        #kernel = torch.reshape(window, [1, 1, size, size]).repeat(3, 1, 1, 1)
        kernel = torch.reshape(window, [1, 1, size, size])
        return kernel

    def forward(self, img0, img1):
        """
        :param img0: range in (-1, 1)
        :param img1: range in (-1, 1)
        :return: SSIM loss i.e. 1 - ssim
        """
        mu0 = torch.nn.functional.conv2d(img0, self.kernel, padding=0, groups=1)
        mu1 = torch.nn.functional.conv2d(img1, self.kernel, padding=0, groups=1)
        mu0_sq = torch.pow(mu0, 2)
        mu1_sq = torch.pow(mu1, 2)
        var0 = torch.nn.functional.conv2d(img0 * img0, self.kernel, padding=0, groups=1) - mu0_sq
        var1 = torch.nn.functional.conv2d(img1 * img1, self.kernel, padding=0, groups=1) - mu1_sq
        covar = torch.nn.functional.conv2d(img0 * img1, self.kernel, padding=0, groups=1) - mu0 * mu1
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        ssim_numerator = (2 * mu0 * mu1 + c1) * (2 * covar + c2)
        ssim_denominator = (mu0_sq + mu1_sq + c1) * (var0 + var1 + c2)
        ssim = ssim_numerator / ssim_denominator
        ssim_loss = 1.0 - ssim
        # print('ssim_loss', ssim_loss)
        return ssim_loss


class MixedPix2PixLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPix2PixLoss, self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred, target):

        ssim_loss = torch.mean(self.ssim_loss(pred, target))
        l1_loss = self.l1_loss(pred, target)
        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss

class MixedPix2PixLoss_mask(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPix2PixLoss_mask, self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss(reduction = 'none')
        print('loss: ssim-', alpha, 'l1-', 1 - alpha)

    def forward(self, pred, target, mask=None):

        if mask != None and torch.count_nonzero(mask).item() != 0:
            ssim_loss = self.ssim_loss(pred*mask, target*mask)
            ssim_loss = torch.sum(ssim_loss) / torch.count_nonzero(mask)

            l1_loss = self.l1_loss(pred*mask, target*mask)
            l1_loss = torch.sum(l1_loss) / torch.count_nonzero(mask)
        else:
            ssim_loss = torch.mean(self.ssim_loss(pred, target))
            l1_loss = torch.mean(self.l1_loss(pred, target))

        weighted_mixed_loss = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss
        return weighted_mixed_loss


if __name__ == '__main__':
    # hole_area_fake = gen_hole_area((64, 64),(272, 272))
    # print(hole_area_fake)
    # mask = gen_input_mask(
    #     shape=(4, 1, 272, 272),
    #     hole_size=((64, 64),(64, 64)),
    #     hole_area=None,
    #     max_holes=1)
    #
    # print(mask.shape)
    #
    # plt.figure(figsize=(9, 3), dpi=300, tight_layout=True)
    # plt.subplot(1, 3, 1)
    # plt.imshow(mask[0].detach().cpu().squeeze().numpy(), cmap="gray")
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask[1].detach().cpu().squeeze().numpy(), cmap="gray")
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask[2].detach().cpu().squeeze().numpy(), cmap="gray")
    # plt.show()
    # plt.clf()
    # plt.close()
    import cv2 as cv
    # x1 = torch.FloatTensor(1, 1, 272, 272)
    # x2 = torch.FloatTensor(1, 1, 272, 272)
    source_path = '/mnt/test_noisy/GL.jpg'
    to_range_norm = lambda x: 2.*x - 1.

    img = cv.imread(source_path, 0)
    blur1 = cv.GaussianBlur(img,(13, 13),0)

    img = img/ 255.0
    img = to_range_norm(img)
    img = toTensor(img).to(torch.float32)

    blur1 = blur1/ 255.0
    blur1 = to_range_norm(blur1)
    blur1 = toTensor(blur1).to(torch.float32)

    plt.figure(figsize=(6, 3), dpi=300, tight_layout=True)
    plt.subplot(1, 2, 1)
    plt.imshow(img.detach().cpu().squeeze().numpy(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(blur1.detach().cpu().squeeze().numpy(), cmap="gray")
    plt.show()

    # edgloss = EDGLoss()
    # loss = edgloss(x1, x1)
    joint_loss = JointLoss()
    loss = joint_loss(blur1, img)
    print(loss)
