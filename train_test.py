import os.path
from torch import optim
from utils import *
from dataset import *
import matplotlib.pyplot as plt
from network import *
from tqdm import tqdm
import wandb
from image_metrics import *
import numpy as np

synthrad_metrics_stage1 = ImageMetrics(dynamic_range=[-150., 850.])
synthrad_metrics = ImageMetrics(dynamic_range=[-1024., 3000.])

set_seed_torch(14)
def post_fun(out, ct, lct, mask, min_v=-1024, max_v=3000):

    out = out.detach().cpu().squeeze().numpy()
    ct = ct.detach().cpu().squeeze().numpy()
    lct = lct.squeeze().numpy()
    mask = mask.detach().cpu().squeeze().numpy()

    out = out * (max_v - min_v) + min_v
    ct = ct * (max_v - min_v) + min_v

    out = np.clip(out, min_v, max_v)
    ct = np.clip(ct, min_v, max_v)

    out = out[int(lct[1]):int(lct[1]) + int(lct[3]), int(lct[0]):int(lct[0]) + int(lct[2])]
    ct = ct[int(lct[1]):int(lct[1]) + int(lct[3]), int(lct[0]):int(lct[0]) + int(lct[2])]
    mask = mask[int(lct[1]):int(lct[1]) + int(lct[3]), int(lct[0]):int(lct[0]) + int(lct[2])]

    return out, ct, mask

def train(data_loader_train, epoch, interval=1):

    loss_gbs = []
    stage1.train()
    stage2.train()
    resbranch.train()

    if epoch <= epoch_stage1:
        print("stage first {}/{} epoch".format(epoch, epoch_stage1))
    elif epoch <= epoch_stage2:
        print("stage second {}/{} epoch".format(epoch, epoch_stage2))
    else:
        print("stage total {}/{} epoch".format(epoch, epoch_total))

    for iteration, (samp_img, _) in enumerate(tqdm(data_loader_train)):

        if iteration % interval != 0:
            continue

        samp_img_gpu = samp_img.to(device)
        (samp_mr_gpu, samp_ct_gpu, samp_ct2_gpu, mask_gpu) = torch.split(samp_img_gpu, [5, 1, 1, 1], dim=1)

        if epoch <= epoch_stage1:

            optimizer_stage1.zero_grad()
            out_global = stage1(samp_mr_gpu*mask_gpu)
            loss_gb = 0.95 * criterion(out_global, samp_ct2_gpu, mask_gpu) + 0.05 * criterion(out_global, samp_ct2_gpu,
                                                                                              1 - mask_gpu)
            loss_gbs.append(loss_gb.item())
            loss_gb.backward()
            optimizer_stage1.step()

        elif epoch <= epoch_stage2:

            optimizer_stage2.zero_grad()
            out_global = stage1(samp_mr_gpu*mask_gpu)
            out_global_cp = out_global.clone().detach()
            out_second = stage2(out_global_cp*mask_gpu)
            loss_gb2 = 0.95 * criterion(out_second, samp_ct_gpu, mask_gpu) + 0.05 * criterion(out_second, samp_ct_gpu,
                                                                                              1 - mask_gpu)
            loss_gbs.append(loss_gb2.item())
            loss_gb2.backward()
            optimizer_stage2.step()
        else:

            optimizer_stage1.zero_grad()
            optimizer_stage2.zero_grad()
            optimizer_resbranch.zero_grad()

            out_global = stage1(samp_mr_gpu*mask_gpu)
            out_global_cp = out_global.clone().detach()
            out_second = stage2(out_global_cp*mask_gpu)
            out_third = resbranch(samp_mr_gpu*mask_gpu)
            out = out_second + out_third

            loss_gb3 = 0.95 * criterion(out, samp_ct_gpu, mask_gpu) + 0.05 * criterion(out, samp_ct_gpu, 1 - mask_gpu)
            loss_gbs.append(loss_gb3.item())
            loss_gb3.backward()
            optimizer_resbranch.step()

    loss_gbs_v = np.sum(loss_gbs)
    print('train epoch:', epoch, 'loss: ', loss_gbs_v)

    torch.save({
        'epoch': epoch,
        'model_stage1': stage1.state_dict(),
        'model_stage2': stage2.state_dict(),
        'model_resbranch': resbranch.state_dict(),
        'loss': loss_gbs_v
    }, os.path.join(model_path, 'model_{}.pth'.format(epoch)))

    torch.save({
        'epoch': epoch,
        'model_stage1': stage1.state_dict(),
        'model_stage2': stage2.state_dict(),
        'model_resbranch': resbranch.state_dict(),
        'optimizer_stage1': optimizer_stage1.state_dict(),
        'optimizer_stage2': optimizer_stage2.state_dict(),
        'optimizer_resbranch': optimizer_resbranch.state_dict(),
        'loss': loss_gbs_v
    },  os.path.join(model_path, 'last.pth'.format(epoch)))

    print('** save epoch ', epoch, '.')
    log_str = " train epoch:%d loss_gb:%f" % (epoch, loss_gbs_v)
    logger.info(log_str)
    return loss_gbs_v

def test(data_loader_test, epoch):

    stage1.eval()
    stage2.eval()
    resbranch.eval()

    ds_len = len(data_loader_test)
    metrics = np.zeros((1, 3, ds_len))

    with torch.no_grad():
        for iteration, (samp_img, samp_lct) in enumerate(tqdm(data_loader_test)):

            samp_img_gpu = samp_img.to(device)
            (samp_mr_gpu, samp_ct_gpu, samp_ct2_gpu, mask_gpu) = torch.split(samp_img_gpu, [5, 1, 1, 1], dim=1)

            out_global = stage1(samp_mr_gpu*mask_gpu)

            if epoch <= epoch_stage1:

                out_cal, ct_cal, mask_cal = post_fun(out_global, samp_ct2_gpu, samp_lct, mask_gpu, min_v=-150, max_v=850)
                m_global = synthrad_metrics_stage1.score_patient(ct_cal, out_cal, mask_cal)

                if iteration % 300 == 0:
                    show_mr = samp_mr_gpu.detach().cpu().squeeze().numpy()
                    show_ct2 = samp_ct2_gpu.detach().cpu().squeeze().numpy()
                    show_mask = mask_gpu.detach().cpu().squeeze().numpy()
                    show_out1 = out_global.detach().cpu().squeeze().numpy()

                    plt.figure(figsize=(9, 3), dpi=100, tight_layout=True)
                    plt.subplot(1, 3, 1)
                    plt.imshow(show_mr[2], cmap="gray")
                    plt.title('s1 psnr: {:.4f} ssim: {:.4f} mae:{:.4f}'.format(m_global['psnr'], m_global['ssim'], m_global['mae']))
                    plt.subplot(1, 3, 2)
                    plt.imshow(show_ct2, cmap="gray")
                    plt.subplot(1, 3, 3)
                    plt.imshow(show_out1*show_mask, cmap="gray")
                    plt.savefig("visualization/s1_epoch{}_{}.jpg".format(epoch, iteration), dpi=300)
                    plt.clf()
                    plt.close()

                metrics[0, 0, iteration] = m_global['psnr']
                metrics[0, 1, iteration] = m_global['ssim']
                metrics[0, 2, iteration] = m_global['mae']

            elif epoch <= epoch_stage2:

                out_second = stage2(out_global*mask_gpu)
                out_cal, ct_cal, mask_cal = post_fun(out_second, samp_ct_gpu, samp_lct, mask_gpu)
                m_global = synthrad_metrics.score_patient(ct_cal, out_cal, mask_cal)

                if iteration % 300 == 0:
                    show_mr = samp_mr_gpu.detach().cpu().squeeze().numpy()
                    show_ct2 = samp_ct2_gpu.detach().cpu().squeeze().numpy()
                    show_mask = mask_gpu.detach().cpu().squeeze().numpy()
                    show_out1 = out_global.detach().cpu().squeeze().numpy()

                    show_ct  = samp_ct_gpu.detach().cpu().squeeze().numpy()
                    show_out2 = out_second.detach().cpu().squeeze().numpy()

                    plt.figure(figsize=(15, 3), dpi=300, tight_layout=True)
                    plt.subplot(1, 5, 1)
                    plt.imshow(show_mr[2], cmap="gray")
                    plt.subplot(1, 5, 2)
                    plt.imshow(show_ct, cmap="gray")
                    plt.title('s2 psnr: {:.4f} ssim: {:.4f} mae:{:.4f}'.format(m_global['psnr'], m_global['ssim'], m_global['mae']))
                    plt.subplot(1, 5, 3)
                    plt.imshow(show_out1*show_mask, cmap="gray")
                    plt.subplot(1, 5, 4)
                    plt.imshow(show_ct2, cmap="gray")
                    plt.subplot(1, 5, 5)
                    plt.imshow(show_out2*show_mask, cmap="gray")
                    plt.savefig("visualization/s2_epoch{}_{}.jpg".format(epoch, iteration), dpi=300)
                    plt.clf()
                    plt.close()

                metrics[0, 0, iteration] = m_global['psnr']
                metrics[0, 1, iteration] = m_global['ssim']
                metrics[0, 2, iteration] = m_global['mae']

            else:

                out_second = stage2(out_global*mask_gpu)
                out_third = resbranch(samp_mr_gpu * mask_gpu)
                out = out_second + out_third

                out_cal, ct_cal, mask_cal = post_fun(out, samp_ct_gpu, samp_lct, mask_gpu)
                m_global = synthrad_metrics.score_patient(ct_cal, out_cal, mask_cal)

                if iteration % 300 == 0:
                    show_mr = samp_mr_gpu.detach().cpu().squeeze().numpy()
                    show_ct2 = samp_ct2_gpu.detach().cpu().squeeze().numpy()
                    show_mask = mask_gpu.detach().cpu().squeeze().numpy()
                    show_out1 = out_global.detach().cpu().squeeze().numpy()

                    show_ct = samp_ct_gpu.detach().cpu().squeeze().numpy()
                    show_out2 = out_second.detach().cpu().squeeze().numpy()

                    plt.figure(figsize=(15, 3), dpi=300, tight_layout=True)
                    plt.subplot(1, 5, 1)
                    plt.imshow(show_mr[2], cmap="gray")
                    plt.subplot(1, 5, 2)
                    plt.imshow(show_ct, cmap="gray")
                    plt.title('s3 psnr: {:.4f} ssim: {:.4f} mae:{:.4f}'.format(m_global['psnr'], m_global['ssim'],
                                                                               m_global['mae']))
                    plt.subplot(1, 5, 3)
                    plt.imshow(show_out1 * show_mask, cmap="gray")
                    plt.subplot(1, 5, 4)
                    plt.imshow(show_ct2, cmap="gray")
                    plt.subplot(1, 5, 5)
                    plt.imshow(show_out2 * show_mask, cmap="gray")
                    plt.savefig("visualization/s3_epoch{}_{}.jpg".format(epoch, iteration), dpi=300)
                    plt.clf()
                    plt.close()

                metrics[0, 0, iteration] = m_global['psnr']
                metrics[0, 1, iteration] = m_global['ssim']
                metrics[0, 2, iteration] = m_global['mae']

        print('psnr:{} ssim:{} mae:{}'.format(np.nanmean(metrics[0][0]), np.nanmean(metrics[0][1]), np.nanmean(metrics[0][2])))
        log_str = "test epoch:%d psnr %f ssim:%f mae:%f " % (epoch, np.nanmean(metrics[0][0]), np.nanmean(metrics[0][1]), np.nanmean(metrics[0][2]))
        logger.info(log_str)

    return {'psnr': np.nanmean(metrics[0][0]), 'ssim': np.nanmean(metrics[0][1]), 'mae': np.nanmean(metrics[0][2])}

if __name__ == '__main__':

    config = {
        'anatomy': 'brain', # 'brain' or 'pelvis
        'resume': False,
        'wandb': False,
        'iftrain': True,
        'iftest': True,
        'project_name': 'synthRAD_MR_to_CT',
        'epoch_stage1': 100,
        'epoch_stage2': 200,
        'epoch_total': 300,
        'batch_size': 4,
        'device_num': '0',
        'learning_rate': 0.0001,
        'model_path': 'checkpoint',
        'last_checkpoint_name': 'checkpoint/last.pth',
        'dataset_path': ['synthRAD_interval_2_brain_train.npz',
                         'synthRAD_interval_1_brain_test.npz',
                         'synthRAD_interval_2_pelvis_train.npz',
                         'synthRAD_interval_1_pelvis_test.npz'],
        'log_path': 'log',
        'visual_path': 'visualization'
    }

    anatomy = config['anatomy']

    resume = config['resume']
    ifwandb = config['wandb']
    iftrain = config['iftrain']
    iftest = config['iftest']

    projectname = config['project_name']
    epoch_stage1 = config['epoch_stage1']
    epoch_stage2 = config['epoch_stage2']
    epoch_total = config['epoch_total']

    batch_size = config['batch_size']
    device = 'cuda:' + config['device_num']
    learning_rate = config['learning_rate']
    model_path = config['model_path']
    last_checkpoint_name = config['last_checkpoint_name']

    if anatomy == 'brain':
        dataset_train_path = config['dataset_path'][0]
        dataset_test_path = config['dataset_path'][1]
    else:
        dataset_train_path = config['dataset_path'][2]
        dataset_test_path = config['dataset_path'][3]

    log_path = config['log_path']
    visual_path = config['visual_path']
    last_epoch = 0
    last_loss = 0

    if ifwandb:
        assert wandb.run is None
        run = wandb.init(project=projectname, config=config)
        assert wandb.run is not None
        print('config:', wandb.config)

    stage1 = MyUNet_plus(32).to(device)
    stage2 = MyUNet(32).to(device)
    resbranch = MyUNet_plus(32, act=False).to(device)

    optimizer_stage1 = optim.Adam(stage1.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0001)
    optimizer_stage2 = optim.Adam(stage2.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0001)
    optimizer_resbranch = optim.Adam(resbranch.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0001)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = get_logger(log_path)
    if not os.path.exists(visual_path):
        os.mkdir(visual_path)

    if resume:
        if not os.path.exists(last_checkpoint_name):
            print('no last checkpoint, start a new train.')
        else:
            checkpoint = torch.load(last_checkpoint_name)

            stage1.load_state_dict(checkpoint['model_stage1'])
            stage2.load_state_dict(checkpoint['model_stage2'])
            resbranch.load_state_dict(checkpoint['model_resbranch'])

            optimizer_stage1.load_state_dict(checkpoint['optimizer_stage1'])
            optimizer_stage2.load_state_dict(checkpoint['optimizer_stage2'])
            optimizer_resbranch.load_state_dict(checkpoint['optimizer_resbranch'])

            last_epoch = checkpoint['epoch']
            last_loss = checkpoint['loss']
            print('load checkpoint from epoch {} loss:{}'.format(last_epoch, last_loss))


    if iftrain:
        print('train...')
        dataset_train = CreateDataset_npz(dataset_path=dataset_train_path)
        data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        sampler=None,
                                                        drop_last=True)


    if iftest:
        print('test...')
        dataset_test = CreateDataset_npz(dataset_path=dataset_test_path)
        data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       sampler=None,
                                                       drop_last=True)

    criterion = MixedPix2PixLoss_mask(alpha=0.5).to(device)

    for epoch in range(last_epoch + 1, epoch_total + 1):
        if iftrain:
            train_loss = train(data_loader_train, epoch, interval=2)

            if iftest and epoch%2==0:
                test_metrics = test(data_loader_test, epoch)
                print(test_metrics)
                if ifwandb and iftrain and iftest:
                    wandb.log({"train loss": train_loss,
                               "psnr": test_metrics['psnr'],
                               "ssim": test_metrics['ssim'],
                               "mae": test_metrics['mae']
                               })

        if iftest and not iftrain:
            test_metrics = test(data_loader_test, epoch)
            print(test_metrics)
            break

