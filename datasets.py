# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import torch
import random
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torch.utils.data import Dataset
from PIL import Image


class LidarImageDetectionDataset(Dataset):
    def __init__(self, root, transform=None, lidar_transform=None, img_size=(1280, 960), shuffle_dataset=True):
        self.root = root
        self.transform = transform
        self.lidar_transform = lidar_transform
        self.img_size = img_size

        # 存储所有数据的路径
        self.data_paths = []
        for folder in os.listdir(root):
            original_folder = os.path.join(root, folder, 'output_images', 'original')
            lidar_folder = os.path.join(root, folder, 'output_images', 'lidar')
            labels_folder = os.path.join(root, folder, 'output_images', 'labels')

            if os.path.exists(original_folder):
                for img_file in os.listdir(original_folder):
                    if img_file.endswith('_original.png'):
                        base_name = img_file.replace('_original.png', '')
                        original_path = os.path.join(original_folder, img_file)
                        lidar_path = os.path.join(lidar_folder, base_name + '_lidar.png')
                        label_path = os.path.join(labels_folder, base_name + '.txt')

                        if os.path.exists(lidar_path) and os.path.exists(label_path):
                            self.data_paths.append((original_path, lidar_path, label_path))

        # ✅ 可选打乱
        if shuffle_dataset:
            random.shuffle(self.data_paths)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        original_path, lidar_path, label_path = self.data_paths[idx]

        original_img = Image.open(original_path).convert("RGB").resize(self.img_size)
        lidar_img = Image.open(lidar_path).convert("L").resize(self.img_size)

        if self.transform:
            original_img = self.transform(original_img)
        if self.lidar_transform:
            lidar_img = self.lidar_transform(lidar_img)
        else:
            lidar_img = transforms.ToTensor()(lidar_img)

        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())
                boxes.append([cx, cy, w, h])
                labels.append(int(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {'labels': labels, 'boxes': boxes}

        return original_img, lidar_img, target

    @staticmethod
    def collate_fn(batch):
        original_imgs, lidar_imgs, targets = zip(*batch)
        original_imgs = torch.stack(original_imgs)
        lidar_imgs = torch.stack(lidar_imgs)
        return (original_imgs, lidar_imgs), list(targets)

# class LidarImageDetectionDataset(Dataset):
#     def __init__(self, root, transform=None, lidar_transform=None, img_size=(1280, 960)):
#         self.root = root
#         self.transform = transform
#         self.lidar_transform = lidar_transform
#         self.img_size = (1280, 960)
#
#         # 存储所有数据的路径
#         self.data_paths = []
#         for folder in os.listdir(root):
#             original_folder = os.path.join(root, folder, 'output_images', 'original')
#             lidar_folder = os.path.join(root, folder, 'output_images', 'lidar')
#             labels_folder = os.path.join(root, folder, 'output_images', 'labels')
#
#             if os.path.exists(original_folder):
#                 for img_file in os.listdir(original_folder):
#                     if img_file.endswith('_original.png'):
#                         base_name = img_file.replace('_original.png', '')
#                         original_path = os.path.join(original_folder, img_file)
#                         lidar_path = os.path.join(lidar_folder, base_name + '_lidar.png')
#                         label_path = os.path.join(labels_folder, base_name + '.txt')
#
#                         if os.path.exists(lidar_path) and os.path.exists(label_path):
#                             self.data_paths.append((original_path, lidar_path, label_path))
#
#     def __len__(self):
#         return len(self.data_paths)
#
#     def __getitem__(self, idx):
#         original_path, lidar_path, label_path = self.data_paths[idx]
#
#         # 加载原始图片和激光雷达图片
#         original_img = Image.open(original_path).convert("RGB").resize(self.img_size)
#         lidar_img = Image.open(lidar_path).convert("L").resize(self.img_size)
#
#         if self.transform:
#             original_img = self.transform(original_img)
#         if self.lidar_transform:
#             lidar_img = self.lidar_transform(lidar_img)
#         else:
#             lidar_img = transforms.ToTensor()(lidar_img)
#
#         # 加载标签
#         boxes, labels = [], []
#         with open(label_path, 'r') as f:
#             for line in f.readlines():
#                 cls, cx, cy, w, h = map(float, line.strip().split())
#                 boxes.append([cx, cy, w, h])
#                 labels.append(int(cls))
#
#         boxes = torch.tensor(boxes, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.long)
#
#         target = {'labels': labels, 'boxes': boxes}
#
#         return original_img, lidar_img, target
#
#     @staticmethod
#     def collate_fn(batch):
#         original_imgs, lidar_imgs, targets = zip(*batch)
#         original_imgs = torch.stack(original_imgs)
#         lidar_imgs = torch.stack(lidar_imgs)
#         return (original_imgs, lidar_imgs), list(targets)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'LIDAR_IMG_DET':
        dataset = LidarImageDetectionDataset(
            root=args.data_path,
            transform=transform,
            lidar_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 示例的雷达图归一化，可自定义
            ]),
            # img_size=(args.input_size, args.input_size),
            img_size=(1280, 960),
            shuffle_dataset = True
        )
        nb_classes = args.nb_classes
    elif args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


'''
###################
去掉 这个会resize
###################
'''


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform
#
#     t = []
#     if resize_im:
#         size = int(args.input_size / args.eval_crop_ratio)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))
#
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)

def build_transform(is_train, args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return transform


# def build_transform(is_train, args):
#     img_size = args.img_size  # (width, height)
#     lidar_size = args.lidar_size  # (width, height)
#
#     if is_train:
#         transform_img = transforms.Compose([
#             transforms.Resize((img_size[1], img_size[0])),  # (height, width)
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#         ])
#
#         transform_lidar = transforms.Compose([
#             transforms.Resize((lidar_size[1], lidar_size[0])),  # (height, width)
#             transforms.ToTensor(),
#         ])
#         return transform_img, transform_lidar
#
#     transform_img = transforms.Compose([
#         transforms.Resize((img_size[1], img_size[0])),  # (height, width)
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#     ])
#
#     transform_lidar = transforms.Compose([
#         transforms.Resize((lidar_size[1], lidar_size[0])),  # (height, width)
#         transforms.ToTensor(),
#     ])
#
#     return transform_img, transform_lidar

