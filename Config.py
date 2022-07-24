# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 9:20 
# @Author : yc096
# @File : Config.py
import os

# -----Project Setting-----#
PROJECT_ROOT = r'C:\WorkSpace\Polyp-Segmentation-Networks'
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'Data')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'CheckPoint')
IMAGE_ROOT = os.path.join(PROJECT_ROOT,'Image')

# -----Train Config-----#
START_EPOCH = 1
MAX_EPOCH = 10
TEST_PER_EPOCH = 1
SAVE_MODEL_PER_EPOCH = 1
TRAIN_SIZE_H, TRAIN_SIZE_W = 352, 352

# -----Dataloader Config-----#
DATASETS_NAME_TRAIN = ['Kvasir-CVC-ClinicDB']
DATASETS_NAME_TEST = ['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','CVC-300','EndoTect-2020']
BATCH_SIZE = 16
NUM_WORERS = 8
