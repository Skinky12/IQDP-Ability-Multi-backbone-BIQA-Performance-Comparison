import random
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
import os.path
import numpy as np
import scipy.io as scio
from PIL import Image
import os
import torch
import torchvision
import pandas as pd
import scipy
from torchvision.transforms import functional as F



class KONIQFolder(data.Dataset):

    def __init__(self, root, index, transform):
        data = pd.read_csv(os.path.join(root, 'koniq10k_scores_and_distributions.csv'))
        labels = np.array(data['MOS_zscore']).astype(np.float32)
        imname = np.array(data['image_name']).tolist()

        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        labels = labels.tolist()
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, '1024x768', imname[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        length = len(self.samples)
        return length

# class LIVEChallengeFolder(data.Dataset):
#
#     def __init__(self, root, index, transform):
#
#         imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
#         imgpath = imgpath['AllImages_release']
#         imgpath = imgpath[7:1169] 
#         labels = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
#         labels = labels['AllMOS_release'].astype(np.float32)
#         labels = labels[0][7:1169]
#         labels = (labels - np.min(labels))/(np.max(labels) - np.min(labels))
#         sample = []
#         for i, item in enumerate(index):
#             sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))
#
#         self.samples = sample
#         self.transform = transform
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, label = self.samples[index]
#         img = pil_loader(path)
#         img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         length = len(self.samples)
#         return length




class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169] 
        labels = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = labels['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]
        labels = (labels - np.min(labels))/(np.max(labels) - np.min(labels))
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)
        _, height, width = image.shape
        max_size = 512
        if max(height, width) > max_size:
            scale_factor = max_size / max(height, width)
            new_size = (int(height * scale_factor), int(width * scale_factor))
            resized_image = F.resize(image, new_size, antialias=True)
        else:
            resized_image = image
            new_size = (height, width)

        padded_image = torch.full((3, max_size, max_size), 0).float()
        left_pad = (max_size - new_size[1]) // 2  
        top_pad = (max_size - new_size[0]) // 2  
        padded_image[:, top_pad:top_pad + new_size[0], left_pad:left_pad + new_size[1]] = resized_image

        padded_image = self.transform(padded_image) 

        return padded_image, label

    def __len__(self):
        length = len(self.samples)
        return length



class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform):
        df = pd.read_csv(os.path.join(root, 'tidmos.csv'), header=None)
        labels = np.array(df[0]).astype(np.float32)
        imname = df[1].tolist()
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        labels = labels.tolist()
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'distorted_images', imname[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        length = len(self.samples)
        return length

class KADIDFolder(data.Dataset):

    def __init__(self, root, index, transform):
        imname = scipy.io.loadmat(os.path.join(root, 'fnames.mat'))['fnames']
        labels = scipy.io.loadmat(os.path.join(root, 'mos.mat'))['mos'][:, 0]
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        labels = labels.tolist()
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'Images', imname[0][item][0]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        length = len(self.samples)
        return length



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DataLoaderIQA(object):
    
    def __init__(self, dataset, path, img_indx, batch_size=1, istrain=True, pre_proc='resize1'):
        proc_funcs = {
            'koniq': KONIQFolder,
            'livec': LIVEChallengeFolder,
            'tid': TID2013Folder,
            'kadid': KADIDFolder,
        }

        resize_size =(384, 512)

        crop_size = (384, 384)

        self.batch_size = batch_size
        self.istrain = istrain

        # cnn
        if pre_proc == 'resize1':
            if istrain:
                if dataset == 'koniq':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        #                              std=(0.26862954, 0.26130258, 0.27577711))])
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))])

                elif dataset == 'tid' or dataset == 'kadid':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                    # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                    #                                  std=(0.5, 0.5, 0.5))])

                elif dataset == 'livec':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                    # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                    #                                  std=(0.5, 0.5, 0.5))])
            else:
                if dataset == 'koniq':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        #                              std=(0.26862954, 0.26130258, 0.27577711))])
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))])

                elif dataset == 'tid' or dataset == 'kadid':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                     std=(0.26862954, 0.26130258, 0.27577711))])
                        # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        #                                  std=(0.5, 0.5, 0.5))])

                elif dataset == 'livec':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                        # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        #                                  std=(0.5, 0.5, 0.5))])


        # transform
        if pre_proc == 'resize2':
            if istrain:
                if dataset == 'koniq':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        #                                  std=(0.26862954, 0.26130258, 0.27577711))])
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))])

                elif dataset == 'tid' or dataset == 'kadid':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                    # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                    #                                  std=(0.5, 0.5, 0.5))])

                elif dataset == 'livec':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                    # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                    #                                  std=(0.5, 0.5, 0.5))])
            else:
                if dataset == 'koniq':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        #                              std=(0.26862954, 0.26130258, 0.27577711))])
                        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))])

                elif dataset == 'tid' or dataset == 'kadid':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                     std=(0.26862954, 0.26130258, 0.27577711))])
                        # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        #                                  std=(0.5, 0.5, 0.5))])

                elif dataset == 'livec':
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(resize_size),
                        torchvision.transforms.RandomCrop(crop_size),
                        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                         std=(0.26862954, 0.26130258, 0.27577711))])
                        # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                        #                                  std=(0.5, 0.5, 0.5))])




        self.data = proc_funcs[dataset](root=path, index=img_indx, transform=transforms)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, drop_last=True,
                num_workers=8) 
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)
        return dataloader

# if __name__ == '__main__':
#
#     # indexs = [xx for xx in range(1162)]
#     # root = r'F:\dataset\修改数据集\LIVEC\LIVEC'
#     # iqadata = DataLoaderIQA(dataset='livec', path=root, batch_size=1, is_huffle=False, index=indexs)
#     # indexs = [xx for xx in range(3000)]
#     # root = r'F:\dataset\修改数据集\tid2013'
#     # iqadata = DataLoaderIQA(dataset='tid', path=root, batch_size=1, is_huffle=False, index=indexs)
#     indexs = [xx for xx in range(10125)]
#     root = r'F:\dataset\修改数据集\KADID\KADID'
#     iqadata = DataLoaderIQA(dataset='koniq', path=root, img_indx=[ii for ii in range(200)], batch_size=2, istrain=True)
#     iqaloader = iqadata.get_data()
#     for img, label in iqaloader:
#         plt.imshow(img[0,0,:,:].squeeze().numpy())
#         plt.show()

if __name__ == '__main__':
    path = r'F:\dataset\修改数据集\LIVEC\LIVEC'
    trainset = DataLoaderIQA(dataset='livec', path=path, img_indx=[ii for ii in range(200)], batch_size=2, istrain=False)
    for img, label in trainset.get_data():
        print(img.shape)