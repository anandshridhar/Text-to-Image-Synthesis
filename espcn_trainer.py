#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_images = sorted(glob.glob(lr_dir + '/*.png'))
        self.hr_images = sorted(glob.glob(hr_dir + '/*.png'))
        
        # Debugging: Print the number of images found
        print(f"Found {len(self.lr_images)} LR images")
        print(f"Found {len(self.hr_images)} HR images")

        if len(self.lr_images) == 0 or len(self.hr_images) == 0:
            raise ValueError("No images found in the specified directories.")

        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_images[idx])
        hr_image = Image.open(self.hr_images[idx])
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image

# Define the transformation including resizing
resize_to = (256, 256)  # Set the desired size
transform = transforms.Compose([
    transforms.Resize(resize_to),
    transforms.ToTensor()
])

# Paths to the LR and HR image directories
lr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_LR_bicubic/X3'  # Example for 3x downsampled images
hr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_HR'

# Create the dataset and dataloader
train_dataset = DIV2KDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Example: Check the first batch of images
for lr, hr in train_loader:
    print(lr.shape, hr.shape)
    break


# In[ ]:





# In[ ]:




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

# Define training function
def train_model(model, dataloader, num_epochs=100, lr=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            # Resize the super-resolved image to match the HR size
            sr = nn.functional.interpolate(sr, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), '/groups/ldbrown/t1capstone/kaggle/ESPCN/espcn_model.pth')

# Paths to the LR and HR image directories
lr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_LR_bicubic/X3'  # Example for 4x downsampled images
hr_dir = '/groups/ldbrown/t1capstone/kaggle/ESPCN/DIV2K_train_HR'

# Create dataloader
train_loader = create_dataloader(lr_dir, hr_dir, scale_factor=3, batch_size=32)

# Initialize model
model = ESPCN(scale_factor=3)

# Train the model
train_model(model, train_loader, num_epochs=100, lr=0.001, device='cuda')


# In[ ]:





# In[ ]:




