import os, sys
import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel


class Text2ImageDataset(Dataset):

    def __init__(self, datasetFile, embeddings_file, transform=None, split=0):
        self.datasetFile = datasetFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))
        self.embeddings = np.load(embeddings_file, allow_pickle=True).item()


    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        # pdb.set_trace()

        right_image = bytes(np.array(example['img']))
        #right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        right_embed = self.embeddings[example_name]


  
        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_images': torch.FloatTensor(wrong_image),
                'inter_embed': torch.FloatTensor(inter_embed),
                'txt': np.array(example['txt']).item().strip().decode('utf-8')
                 }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

# Code added for precomuting XLNet embeddings
    @staticmethod
    def compute_and_save_embeddings(dataset_file, split, output_file, embedding_dim=1024):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased').to(device)
        model.eval()
        embeddings = {}
        with h5py.File(dataset_file, 'r') as f:
            dataset_keys = [str(k) for k in f[split].keys()]
        
        
            with torch.no_grad():
                print('Calculating XLNet Embeddings for {}'.format(split))
                dataset_size = len(dataset_keys)
                for inc, key in enumerate(dataset_keys):
                    print('{}/{}'.format(inc + 1, dataset_size))
                    example = f[split][key]
                    txt = example['txt'][()].strip().decode('utf-8')
                    inputs = tokenizer(txt, return_tensors="pt", padding='max_length', max_length=128).to(device)
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
                    # Pad the embedding to the desired length
                    if embedding.shape[0] < embedding_dim:
                        embedding = np.pad(embedding, (0, embedding_dim - embedding.shape[0]), 'constant')
                    embeddings[key] = embedding
            print('Finished XLNet Embedding calculations')
        np.save(output_file, embeddings)
         
    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        return self.embeddings[example_name]
 


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

