#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

# Define ESPCN model
class ESPCN(nn.Module):
    def __init__(self, scale_factor=3):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, scale_factor**2 * 3, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

# Define DIV2KDataset
class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None):
        self.lr_images = sorted(glob.glob(lr_dir + '/*.png'))
        self.hr_images = sorted(glob.glob(hr_dir + '/*.png'))
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_images[idx])
        hr_image = Image.open(self.hr_images[idx])
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        return lr_image, hr_image

# Define function to create dataloader
def create_dataloader(lr_dir, hr_dir, scale_factor=3, batch_size=16):
    # Transformation for low-resolution images
    lr_transform = transforms.Compose([
        transforms.Resize((256 // scale_factor, 256 // scale_factor)),
        transforms.ToTensor()
    ])
    # Transformation for high-resolution images
    hr_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    # Ensure the HR and generated SR images are resized to match
    common_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = DIV2KDataset(lr_dir=lr_dir, hr_dir=hr_dir, lr_transform=lr_transform, hr_transform=common_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def calculate_metrics(sr, hr):
    sr = sr.cpu().detach().numpy().transpose(0, 2, 3, 1)
    hr = hr.cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    psnr_value = np.mean([psnr(hr[i], sr[i], data_range=sr[i].max() - sr[i].min()) for i in range(hr.shape[0])])

    ssim_values = []
    for i in range(hr.shape[0]):
        # Print image shapes for debugging
         #print(f"HR Image shape: {hr[i].shape}, SR Image shape: {sr[i].shape}")
        
        # Ensure the window size fits within the image dimensions
        win_size = min(hr[i].shape[0], hr[i].shape[1], 7)  # Set to 7 as a safe lower value
        win_size = max(win_size, 3)  # Ensure win_size is at least 3
        #print(f"Using win_size: {win_size}")
        
        # If win_size is still larger than image dimensions, skip SSIM calculation
        if win_size > hr[i].shape[0] or win_size > hr[i].shape[1]:
            print(f"Skipping SSIM calculation for image {i} due to small size.")
            continue
        
        ssim_value = ssim(hr[i], sr[i], channel_axis=-1, data_range=sr[i].max() - sr[i].min(), win_size=win_size)
        ssim_values.append(ssim_value)

    ssim_value = np.mean(ssim_values) if ssim_values else float('nan')
    return psnr_value, ssim_value

def train_model(model, dataloader, num_epochs=50, lr=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    epoch_losses = []
    epoch_psnr_values = []
    epoch_ssim_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            sr = nn.functional.interpolate(sr, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_psnr, batch_ssim = calculate_metrics(sr, hr)
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim

        epoch_loss /= len(dataloader)
        epoch_psnr /= len(dataloader)
        epoch_ssim /= len(dataloader)

        epoch_losses.append(epoch_loss)
        epoch_psnr_values.append(epoch_psnr)
        epoch_ssim_values.append(epoch_ssim)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')

    torch.save(model.state_dict(), '/groups/ldbrown/t1capstone/kaggle/ESPCN/espcn_model.pth')

    return epoch_losses, epoch_psnr_values, epoch_ssim_values

def evaluate_model(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = nn.functional.interpolate(sr, size=(256, 256), mode='bilinear', align_corners=False)
            batch_psnr, batch_ssim = calculate_metrics(sr, hr)
            psnr_values.append(batch_psnr)
            ssim_values.append(batch_ssim)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.nanmean(ssim_values) if ssim_values else float('nan')
    print(f'Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

    return avg_psnr, avg_ssim

def plot_performance_metrics(losses, psnrs, ssims):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, psnrs, label='PSNR', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssims, label='SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Paths to the validation LR and HR image directories
val_lr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_valid_LR_bicubic/X3'
val_hr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_valid_HR'
lr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_LR_bicubic/X3'
hr_dir =  '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_HR'
# Create dataloader for validation set
val_loader = create_dataloader(val_lr_dir, val_hr_dir, scale_factor=3, batch_size=32)

# Create dataloader for training set
train_loader = create_dataloader(lr_dir, hr_dir, scale_factor=3, batch_size=32)

# Initialize and train the model
model = ESPCN(scale_factor=3)
losses, psnrs, ssims = train_model(model, train_loader, num_epochs=50, lr=0.001, device='cuda')

# Evaluate the model
avg_psnr, avg_ssim = evaluate_model(model, val_loader, device='cuda')

# Plot performance metrics
plot_performance_metrics(losses, psnrs, ssims)

# Print final evaluation metrics
print(f'Final Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')


# In[ ]:




