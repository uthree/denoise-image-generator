import torch
import torch.optim as optim
import os
import random
import math

from dataset import *
from denoising_diffusion_pytorch import GaussianDiffusion
from model import init_unet

import torchvision.transforms as transforms

BATCH_SIZE = 16
BATCH_CHUNK = 4
NUM_EPOCH = 500
DATASET_SIZE = 5000
IMAGE_SIZE = 256

ds = ImageDataset(source_dir_pathes=["/mnt/d/local-develop/lineart2image_data_generator/colorized_256x/"], max_len=DATASET_SIZE, size=IMAGE_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists('./model.pt'):
    model = torch.load('./model.pt')
else:
    unet = init_unet(dim=4, dim_mults=[1, 2, 4, 8, 16])
    model = GaussianDiffusion(
        unet,
        image_size = IMAGE_SIZE,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )

aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((-10, 10))], p=0.5),
        transforms.RandomApply([transforms.RandomCrop((round(IMAGE_SIZE * 0.8), round(IMAGE_SIZE * 0.75)))], p=0.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

model.to(device)

bar_epoch = tqdm(total=len(ds) * NUM_EPOCH, position=1)
bar_batch = tqdm(total=len(ds), position=0)

dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.RAdam(model.parameters())

for i in range(NUM_EPOCH):
    for j, img in enumerate(dl):
        N, C, H, W = img.shape
        optimizer.zero_grad()
        img = aug(img)
        img = img.to(device)
        for c in img.chunk(BATCH_CHUNK, dim=0):
            loss = model(c)
            loss.backward()
            tqdm.write(f"Loss: {loss.item()}")
        optimizer.step()

        if j % 200 == 0:
            tqdm.write("Model is saved!")
            torch.save(model, './model.pt')

        bar_epoch.update(N)
        bar_batch.update(N)
    bar_batch.reset()
