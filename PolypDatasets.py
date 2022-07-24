# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 9:21 
# @Author : yc096
# @File : PolypDatasets.py
import os
import cv2
import Config
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from Utils.opencv_transform import ToTensor, Resize, RandomHorizontalFlip, ColorJitter, RandomScale, CopyPaste


class PolypDatasets(Dataset):
    def __init__(self, DATASET_NAME='', DATASET_TYPE='train'):
        self.dataset_name = DATASET_NAME
        self.dataset_type = DATASET_TYPE
        self.dataset_path = os.path.join(Config.DATASET_ROOT, DATASET_NAME)
        self.train_size_h, self.train_size_w = Config.TRAIN_SIZE_H, Config.TRAIN_SIZE_W
        self.image_folder = os.path.join(self.dataset_path, self.dataset_type)
        self.mask_folder = os.path.join(self.dataset_path, self.dataset_type + '_mask')
        self.list_path = os.path.join(self.dataset_path, self.dataset_type + '.txt')
        self.image_list, self.mask_list = self.ReadTXT(self.list_path)
        self.DATA_TO_RESIZE = transforms.Compose(
            [
                Resize(resize=(self.train_size_h, self.train_size_w)),
            ]
        )
        self.DATA_AUGMENT = transforms.Compose(
            [
                RandomHorizontalFlip(p=0.5),
                # CopyPaste(p=0.5),
                RandomScale(scale_retes=[0.75, 1, 1.25]),
                ColorJitter(),
            ]
        )
        self.DATA_TO_TENSOR = transforms.Compose(
            [
                ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_list[index])
        mask_path = os.path.join(self.mask_folder, self.mask_list[index])
        image, mask = self.Read_IMAGE_MASK(image_path, mask_path)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # （0：bg 255：polyp）
        image, mask = self.DATA_TO_RESIZE((image, mask))
        if self.dataset_type in ('train', 'trainval', 'trainvaltest'):
            image, mask = self.DATA_AUGMENT((image, mask))
        image, mask = self.DATA_TO_TENSOR((image, mask))
        return image, mask

    def __len__(self):
        return len(self.image_list)

    def ReadTXT(self, list_path):
        f = open(list_path, mode='r', encoding='utf-8')
        lines = f.readlines()
        imgs = []
        masks = []
        for line in lines:
            img, mask = line.split('\t')
            imgs.append(img.strip())
            masks.append(mask.strip())
        return imgs, masks

    def Read_IMAGE_MASK(self, image_path, mask_path):
        img = cv2.imread(image_path, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        return img, mask


def get_train_data_loader(BATCH_SIZE=Config.BATCH_SIZE, NUM_WORERS=Config.NUM_WORERS):
    data_loader = []
    for dataset_name in Config.DATASETS_NAME_TRAIN:
        data_loader.append(
            DataLoader(
                PolypDatasets(DATASET_NAME=dataset_name, DATASET_TYPE='train'),
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORERS,
                shuffle=True
            )
        )
    return data_loader


def get_test_data_loader(BATCH_SIZE=Config.BATCH_SIZE, NUM_WORERS=Config.NUM_WORERS):
    data_loader = []
    for dataset_name in Config.DATASETS_NAME_TEST:
        data_loader.append(
            DataLoader(
                PolypDatasets(DATASET_NAME=dataset_name, DATASET_TYPE='test'),
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORERS,
                shuffle=False
            )
        )
    return data_loader
