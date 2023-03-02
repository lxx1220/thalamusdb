import torch
import clip
import os
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import time


class CraigslistDataset(Dataset):
    """Craigslist dataset."""

    def __init__(self, img_paths, transform=None):
        """
        Args:
            img_paths (string): List of paths to all images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # label = img_path.split('/')[-1].split('_')[0]
        label = img_path.split('/')[-1][:-4]

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)

    img_dir = 'craigslist/furniture_imgs/'
    # img_dir = 'craigslist/furniture_imgs_twobytwo/'
    img_paths = [img_dir + f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    t = transforms.Compose([
        transforms.ToPILImage()
    ])

    furniture_dataset = CraigslistDataset(img_paths, t)
    print('The nth image in train dataset: ', furniture_dataset[49][0])

    # mnist = MNIST(root=os.path.expanduser("./data"), download=True, train=False)
    # cifar10 = CIFAR10(root=os.path.expanduser("./data"), download=True, train=False)

    dataset = furniture_dataset

    # extract image feature, code borrowed from: https://github.com/openai/CLIP#zero-shot-prediction
    image_features = []
    start = time.time()
    for image, _ in dataset:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
        image_feature /= image_feature.norm()
        image_features.append(image_feature)
    end = time.time()
    print(f'Time: {end - start}')
    image_features = torch.stack(image_features, dim=1).to(device)
    image_features = image_features.squeeze()

    furniture = pd.read_csv('craigslist/furniture.tsv', sep='\t', index_col=0)

    while True:
        query_str = input()
        query = [query_str]  # 'straight line' 'circle' 'three'
        tokenized = clip.tokenize(query).to(device)
        embeddings = model.encode_text(tokenized)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embedding = embeddings.mean(dim=0)
        embedding /= embedding.norm()

        indexes = furniture[furniture.title.str.contains(query_str, case=False)].index.tolist()

        topk = 25
        # logits = (100. * image_features @ embedding).softmax(dim=-1).cpu().detach()
        logits = (100. * image_features @ embedding).cpu().detach()
        topk_idxs = np.argpartition(logits, -topk)[-topk:]
        topk_idxs = topk_idxs[np.argsort(-logits[topk_idxs])]
        if isinstance(furniture_dataset, CraigslistDataset):
            topk_images = []
            for topk_idx in topk_idxs:
                topk_images.append(dataset[topk_idx][0])
        else:
            topk_images = dataset.data[topk_idxs]

        plt.figure(figsize=(5, 5))  # specifying the overall grid size

        for i in range(topk):
            plt.subplot(5, 5, i+1)
            plt.imshow(topk_images[i])
            plt.title("{0:0.5f}".format(logits[topk_idxs[i]]), fontsize=9, y=0.9)
            plt.xticks([])
            plt.yticks([])

        plt.show()



