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
Use docker image
```
docker pull zhangvae/mr_to_ct
nvidia-docker run -itd --shm-size 128g  -v [/your/path/]:/mnt/ [imageid] /bin/bash
```
Pull the code
```
git clone https://github.com/ZhangZhiHao233/MR-to-CT.git
cd MR-to-CT
```
# Dataset
# Train&Test
# Pretrained Models
