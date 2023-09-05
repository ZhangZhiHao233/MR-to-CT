# A Simple Two-stage Residual Network for MR-CT Translation

![image](https://github.com/ZhangZhiHao233/MR-to-CT/blob/main/figure.jpg)

# Dependencies
```
torch==1.8.0
torchvision==0.9.0
wandb==0.15.9
SimpleITK==2.2.1
scipy==1.6.1
tqdm
```
# Installation
[Optional]Use our docker image:
```
docker pull zhangvae/mr_to_ct
nvidia-docker run -itd --shm-size 128g  -v [/your/path/]:/mnt/ [imageid] /bin/bash
```
Clone this repo:
```
git clone https://github.com/ZhangZhiHao233/MR-to-CT.git
cd MR-to-CT
```
# Dataset
We use the dataset from [synthRAD Challenge](https://synthrad2023.grand-challenge.org/Data/).
Data structure is like
```
Brain_Pelvis
-train
|-brain
  |-1BA001
    |-ct.nii.gz
    |-mr.nii.gz
    |-mask.nii.gz
  |- ...
  |-1BA005
    |-ct.nii.gz
    |-mr.nii.gz
    |-mask.nii.gz
|-pelvis
  |-1PA001
    |-ct.nii.gz
    |-mr.nii.gz
    |-mask.nii.gz
  |- ...
  |-1PA004
    |-ct.nii.gz
    |-mr.nii.gz
    |-mask.nii.gz

-test
|-brain
  |- ...
|-pelvis
  |- ...
```
Preprocess the data and convert it to .npz files
```
python dataset.py
```
# Train&Test
```
python train_test.py
```
# Pretrained Models
We have a pretrained model for the translation of brain and pelvis.
Load the checkpoint:
```
#stage1 = MyUNet_plus(32).to(device)
#stage2 = MyUNet(32).to(device)
#resbranch = MyUNet_plus(32, act=False).to(device)

checkpoint = torch.load(last_checkpoint_name)
stage1.load_state_dict(checkpoint['model_stage1'])
stage2.load_state_dict(checkpoint['model_stage2'])
resbranch.load_state_dict(checkpoint['model_resbranch'])

last_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']
print('load checkpoint from epoch {} loss:{}'.format(last_epoch, last_loss))
```
