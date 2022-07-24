# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 10:02 
# @Author : yc096
# @File : train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.loss import structure_loss
from Utils.Trainer import Trainer
from Model.LPDNet import LPDNet

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LPDNet().to(device)
    criterion = structure_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    lr_updater = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    trainer = Trainer(model,optimizer,criterion,lr_updater,device)
    trainer.train()