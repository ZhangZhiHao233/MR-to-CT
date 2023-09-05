import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
import os
from tqdm import tqdm
from skimage.transform import resize


dataser_path = '../../dataset'
MASK_POINT = []
MASK_AREA = []

def generate_mask(img_size, padding, val=False):

    mask = np.ones(img_size)
    # print('mask size: ', mask.shape)

    mask_size_w = random.randint(img_size[0]//4, img_size[0]-1)
    mask_size_h = random.randint(img_size[1]//4, img_size[1]-1)
    # mask_size = random.randint(img_size//16, img_size//4)

    if (mask_size_w == img_size[0]-1 and mask_size_h == img_size[1]-1) \
            or val==True:
        mask = np.expand_dims(1-mask, axis=0)
        mask, location = mypadding(mask, x=padding[0],y=padding[1], v=1)
        return mask

    c_x = random.randint(0, img_size[0]-1)
    c_y = random.randint(0, img_size[1]-1)

    box_l_x = c_x-mask_size_w//2
    box_l_y = c_y-mask_size_h//2
    box_r_x = c_x+mask_size_w//2
    box_r_y = c_y+mask_size_h//2

    if box_l_x < 0:
        box_l_x = 0
    if box_l_y < 0:
        box_l_y = 0
    if box_r_x > img_size[0]-1:
        box_r_x = img_size[0]-1
    if box_r_y > img_size[1]-1:
        box_r_y = img_size[1]-1

    mask[box_l_y:box_r_y, box_l_x:box_r_x] = 0
    # print('*', c_y-mask_size//2, c_y + mask_size, c_x-mask_size//2,c_x + mask_size)
    mask = np.expand_dims(mask, axis=0)
    mask, location = mypadding(mask, x=padding[0],y=padding[1], v=1)

    MASK_POINT.append([c_x+location[0][0], c_y+location[0][1]])
    MASK_AREA.append([box_l_y, box_r_y, box_l_x, box_r_x])
    # print(mask.shape)
    return mask

def getmsks(img_size, channels, padding, val=False):
    msks = []
    for i in range(channels):
        # print('channel: ', i)
        msks.append(generate_mask(img_size, padding, val))
    msks = np.concatenate(msks, axis=0)
    return msks

def normalize(img, type='mr'):
    if type == 'mr':
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
        # img = np.clip(img, 0, 1)
        return img
    else:
        # 后面试试直接 min max，不固定
        min_value = -1024
        max_value = 3000
        img = (img - min_value) / (max_value - min_value)
        # img = np.clip(img, 0, 1)
        return img

def mypadding(img, x=288, y=288, v=0):

    # print(img.shape)
    temp_x = x - img.shape[2]
    padding_x_fore = temp_x // 2
    padding_x_behind = temp_x - padding_x_fore

    temp_y = y - img.shape[1]
    padding_y_fore = temp_y // 2
    padding_y_behind = temp_y - padding_y_fore

    # print(padding_x_fore, padding_x_behind, padding_y_fore, padding_y_behind)
    padded_img = np.pad(img, ((0,0),(padding_y_fore, padding_y_behind), (padding_x_fore, padding_x_behind)), mode='constant',
                            constant_values=v)

    # x, y, w, h
    img_location = np.array([padding_x_fore, padding_y_fore, img.shape[2], img.shape[1]])
    img_location = np.expand_dims(img_location, 0)
    # print(img_location)
    return padded_img, img_location

def cal_min_max(path):

    brain_ds = os.listdir(path)

    size_heights = []
    size_widths = []

    mr_min_values = []
    mr_max_values = []

    ct_min_values = []
    ct_max_values = []

    for i in range(len(brain_ds)):
        bd = brain_ds[i]
        if bd == '.DS_Store':
            continue

        path_temp = os.path.join(path, bd)
        path_mr = os.path.join(path_temp, 'mr.nii.gz')
        path_ct = os.path.join(path_temp, 'ct.nii.gz')

        mr = sitk.ReadImage(path_mr)
        mr = sitk.GetArrayFromImage(mr)

        # print(mr.shape)
        size_heights.append(mr.shape[1])
        size_widths.append(mr.shape[2])
        mr_min_values.append(mr.min())
        mr_max_values.append(mr.max())

        ct = sitk.ReadImage(path_ct)
        ct = sitk.GetArrayFromImage(ct)
        ct_min_values.append(ct.min())
        ct_max_values.append(ct.max())

    print('size')
    print('--height: {}-{}'.format(np.min(size_heights), np.max(size_heights)))
    print('--width: {}-{}'.format(np.min(size_widths), np.max(size_widths)))

    print('mr')
    print('--value: {}-{}'.format(np.min(mr_min_values), np.max(mr_max_values)))

    print('ct')
    print('--value: {}-{}'.format(np.min(ct_min_values), np.max(ct_max_values)))

def cal_min_max_val(path):

    brain_ds = os.listdir(path)

    size_heights = []
    size_widths = []

    mr_min_values = []
    mr_max_values = []

    for i in range(len(brain_ds)):
        bd = brain_ds[i]
        if bd == '.DS_Store':
            continue

        path_temp = os.path.join(path, bd)
        path_mr = os.path.join(path_temp, 'mr.nii.gz')

        mr = sitk.ReadImage(path_mr)
        mr = sitk.GetArrayFromImage(mr)

        # print(mr.shape)
        size_heights.append(mr.shape[1])
        size_widths.append(mr.shape[2])
        mr_min_values.append(mr.min())
        mr_max_values.append(mr.max())

    print('size')
    print('--height: {}-{}'.format(np.min(size_heights), np.max(size_heights)))
    print('--width: {}-{}'.format(np.min(size_widths), np.max(size_widths)))

    print('mr')
    print('--value: {}-{}'.format(np.min(mr_min_values), np.max(mr_max_values)))


def window_transform(ct_array, windowWidth, windowCenter):
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)

    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg

def generate_train_test_dataset(path, padding, p='brain', t='train', interval=3, save_path='dataset'):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ds = os.listdir(path)
    random.shuffle(ds)

    mr_vecs = []
    ct_vecs = []
    enhance_ct_vecs = []
    mask_vecs = []
    lct_vecs = []

    for case_count, bd in tqdm(enumerate(ds)):
        if bd == '.DS_Store':
            continue

        path_temp = os.path.join(path, bd)
        path_mr = os.path.join(path_temp, 'mr.nii.gz')
        path_ct = os.path.join(path_temp, 'ct.nii.gz')
        path_mask = os.path.join(path_temp, 'mask.nii.gz')

        mr = sitk.ReadImage(path_mr)
        mr = sitk.GetArrayFromImage(mr)
        mr_norm = normalize(mr, type='mr')
        mr_padding, img_location = mypadding(mr_norm, padding[0], padding[1])

        ct = sitk.ReadImage(path_ct)
        ct = sitk.GetArrayFromImage(ct)
        ct_norm = normalize(ct, type='ct')
        ct_padding, _ = mypadding(ct_norm, padding[0], padding[1])

        enhance_ct_norm = window_transform(ct, 1000, 350)
        enhance_ct_padding, _ = mypadding(enhance_ct_norm, padding[0], padding[1])

        mask = sitk.ReadImage(path_mask)
        mask = sitk.GetArrayFromImage(mask)
        mask_padding, _ = mypadding(mask, padding[0], padding[1])
        length = len(mr_padding)

        for index in range(length):

            # if index<5 or index >= length-5:
            #     continue

            if index % interval != 0:
                continue

            index_first = index-2
            index_second = index-1
            index_third = index
            index_forth = index+1
            index_fifth = index+2

            if index < 2:
                index_first = 0
                index_second = 0
            if index >= length-2:
                index_forth = length - 1
                index_fifth = length - 1

            mr_first = mr_padding[index_first]
            mr_second = mr_padding[index_second]
            mr_third = mr_padding[index_third]
            mr_forth = mr_padding[index_forth]
            mr_fifth = mr_padding[index_fifth]

            mr_2_5d = np.array([mr_first, mr_second, mr_third, mr_forth, mr_fifth])
            mr_vecs.append(mr_2_5d)

            lct_vecs.append(img_location)
            ct_vecs.append(ct_padding[index])
            enhance_ct_vecs.append(enhance_ct_padding[index])
            mask_vecs.append(mask_padding[index])

    mr_np = np.array(mr_vecs)
    ct_np = np.array(ct_vecs)
    ct_np = ct_np[:, np.newaxis, :, :]
    enhance_ct_np = np.array(enhance_ct_vecs)
    enhance_ct_np = enhance_ct_np[:, np.newaxis, :, :]
    mask_np = np.array(mask_vecs)
    mask_np = mask_np[:, np.newaxis, :, :]
    mask_np = np.float32(mask_np)

    print(mr_np.shape)
    print(ct_np.shape)
    print(enhance_ct_np.shape)
    print(mask_np.shape)

    img_np = np.concatenate((mr_np, ct_np, enhance_ct_np, mask_np), axis=1)
    lct_np = np.concatenate(lct_vecs, axis=0)
    lct_np = np.float32(lct_np)

    print('mr: ')
    print(' min: ', mr_np.min(), ' max: ', mr_np.max())
    print('ct: ')
    print(' min: ', ct_np.min(), ' max: ', ct_np.max())
    print('ct2: ')
    print(' min: ', enhance_ct_np.min(), ' max: ', enhance_ct_np.max())
    print('mask: ')
    print(' min: ', mask_np.min(), ' max: ', mask_np.max())
    print('image:', img_np.shape, img_np.dtype)
    print('location:', lct_np.shape, lct_np.dtype)

    dataset_name = '{}/synthRAD_interval_{}_{}_{}.npz'.format(save_path, interval, p, t)
    np.savez(dataset_name, img=img_np, lct=lct_np)

def generate_ds():

    # generate_train_test_dataset('../dataset/brain_pelvis/train/brain', padding=[288,288], p='brain', t='train', interval=2, save_path='./')
    # generate_train_test_dataset('../dataset/brain_pelvis/test/brain', padding=[288,288], p='brain', t='test', interval=1, save_path='./')

    generate_train_test_dataset('../dataset/brain_pelvis/train/pelvis', padding=[592,416], p='pelvis', t='train', interval=2, save_path='./')
    generate_train_test_dataset('../dataset/brain_pelvis/test/pelvis', padding=[592,416], p='pelvis', t='test', interval=1, save_path='./')

def generate_valid_ds():

    valid_path = os.path.join('../dataset/', 'brain_pelvis_val/pelvis')
    brain_ds = os.listdir(valid_path)

    for i in range(len(brain_ds)):
        bd = brain_ds[i]
        if bd == '.DS_Store':
            continue
        print(i, bd)

        src_img_vecs = []
        mask_img_vecs = []
        lct_vecs = []

        path_temp = os.path.join(valid_path, bd)
        path_mr = os.path.join(path_temp, 'mr.nii.gz')
        path_mask = os.path.join(path_temp, 'mask.nii.gz')

        mr = sitk.ReadImage(path_mr)
        mr = sitk.GetArrayFromImage(mr)
        mr_norm = normalize(mr, type='mr')

        if bd.find('B') > 0:
            mr_padding, img_location = mypadding(mr_norm, x=288, y=288)
        else:
            mr_padding, img_location = mypadding(mr_norm, x=592, y=416)

        mask = sitk.ReadImage(path_mask)
        mask = sitk.GetArrayFromImage(mask)

        if bd.find('B') > 0:
            mask_padding, _ = mypadding(mask, x=288, y=288)
        else:
            mask_padding, _ = mypadding(mask, x=592, y=416)

        length = len(mr_padding)
        for index in range(length):

            index_first = index - 2
            index_second = index - 1
            index_third = index
            index_forth = index + 1
            index_fifth = index + 2

            if index < 2:
                index_first = 0
                index_second = 0
            if index >= length - 2:
                index_forth = length - 1
                index_fifth = length - 1

            mr_first = mr_padding[index_first]
            mr_second = mr_padding[index_second]
            mr_third = mr_padding[index_third]
            mr_forth = mr_padding[index_forth]
            mr_fifth = mr_padding[index_fifth]

            mr_2_5d = np.array([mr_first, mr_second, mr_third, mr_forth, mr_fifth])
            src_img_vecs.append(mr_2_5d)

            lct_vecs.append(img_location)
            mask_img_vecs.append(mask_padding[index])

        src_img_np = np.array(src_img_vecs)
        mask_img_np = np.array(mask_img_vecs)
        mask_img_np = mask_img_np[:, np.newaxis, :, :]
        mask_img_np = np.float32(mask_img_np)

        img_np = np.concatenate((src_img_np, mask_img_np), axis=1)
        lct_np = np.concatenate(lct_vecs, axis=0)
        lct_np = np.float32(lct_np)

        print('mr: ')
        print(' min: ', src_img_np.min(), ' max: ', src_img_np.max())
        print('mask: ')
        print(' min: ', mask_img_np.min(), ' max: ', mask_img_np.max())
        print('image:', img_np.shape, img_np.dtype)
        print('location:', lct_np.shape, lct_np.dtype)

        dataset_name = '{}/{}/best_{}_5to1_new.npz'.format(valid_path, bd, bd)
        np.savez(dataset_name, img=img_np, lct=lct_np)

def CreateDataset_npz(dataset_path):
    ds = np.load(dataset_path)
    img = ds['img']
    lct = ds['lct']
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(img), torch.from_numpy(lct))
    return dataset

def GetDataset_npz(dataset_path):
    ds = np.load(dataset_path)
    img = ds['img']
    lct = ds['lct']
    return img, lct

if __name__ == '__main__':
    generate_ds()
    # generate_valid_ds()

