import torch
import torch.optim as optim
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image

num_iterations = 1000
num_images = 1
image_size = 256
result_dir = "./results/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('./model.pt')

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

model.to(device)
with torch.no_grad():
    for i in range(num_images):
        initial_noise = torch.randn(1, 3, image_size, image_size).to(device)
        img = model.sample(batch_size = 1)
        path = os.path.join(result_dir, f"{i}.jpg")
        pil_img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
        pil_img = pil_img.resize((256, 256))
        pil_img.save(path)
        
