import torch
import torch.optim as optim
import os
import random

from dataset import *
from denoising_diffusion_pytorch import GaussianDiffusion
from model import init_unet

BATCH_SIZE = 16
NUM_EPOCH = 500
DATASET_SIZE = 50000
IMAGE_SIZE = 256

ds = ImageDataset(source_dir_pathes=["/mnt/d/local-develop/lineart2image_data_generator/colorized_256x/"], max_len=DATASET_SIZE, size=IMAGE_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists('./model.pt'):
    model = torch.load('./model.pt')
else:
    unet = init_unet(dim=4, dim_mults=[1, 2, 4, 8, 16, 32])
    model = GaussianDiffusion(
        unet,
        image_size = 256,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )


model.to(device)

MSE = torch.nn.MSELoss()
bar_epoch = tqdm(total=len(ds) * NUM_EPOCH, position=1)
bar_batch = tqdm(total=len(ds), position=0)

dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.RAdam(model.parameters())

for i in range(NUM_EPOCH):
    for j, img in enumerate(dl):
        N, C, H, W = img.shape
        optimizer.zero_grad()
        img = img.to(device)
        loss = model(img)
        loss.backward()
        optimizer.step()

        tqdm.write(f"Loss: {loss.item()}")
        if j % 200 == 0:
            tqdm.write("Model is saved!")
            torch.save(model, './model.pt')

        bar_epoch.update(N)
        bar_batch.update(N)
    bar_batch.reset()
