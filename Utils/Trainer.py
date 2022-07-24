# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 10:05 
# @Author : yc096
# @File : Trainer.py
import os
import time
import torch
import Config
import PolypDatasets
import numpy as np
from Utils.excel_logger import excel_logger
from Utils.metrics import metrics


class Trainer():
    def __init__(self, model, optim, criterion, lr_scheduler, device):
        # 
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        # 
        self.prefix = time.strftime("%Y%m%d-%H_%M_%S", time.localtime())
        self.checkpoint_root = Config.CHECKPOINT_ROOT
        self.log_file_path = os.path.join(self.checkpoint_root, self.prefix + '.xls')
        # 
        self.logger = excel_logger()
        # 
        self.metrics = metrics()
        # 
        self.start_epoch = Config.START_EPOCH
        self.max_epoch = Config.MAX_EPOCH
        self.current_epoch = -1
        self.test_per_epoch = Config.TEST_PER_EPOCH
        self.save_model_per_epoch = Config.SAVE_MODEL_PER_EPOCH
        #
        self.train_data_loader = PolypDatasets.get_train_data_loader()
        self.test_data_loader = PolypDatasets.get_test_data_loader()
        #
        for i in range(len(Config.DATASETS_NAME_TRAIN)):
            self.logger.add_train_sheet(sheet_name=Config.DATASETS_NAME_TRAIN[i])
        for i in range(len(Config.DATASETS_NAME_TEST)):
            self.logger.add_test_sheet(sheet_name=Config.DATASETS_NAME_TEST[i])

    def train(self):
        print('开始训练')
        for self.current_epoch in range(self.start_epoch, self.max_epoch + 1):
            self.before_train_one_epoch()
            self.train_one_epoch()
            self.after_train_one_epoch()
        print('训练结束')

    def before_train_one_epoch(self):
        self.model.train()
        self.metrics.reset()

    def after_train_one_epoch(self):
        self.model.eval()
        self.metrics.reset()

    def train_one_epoch(self):

        for i, data_loader in enumerate(self.train_data_loader):
            self.model.train()
            self.metrics.reset()
            excel = self.logger.work_book.get_sheet(Config.DATASETS_NAME_TRAIN[i])
            epoch_loss = 0.0
            epoch_time = self.getTime()
            for inputs, masks in data_loader:
                self.optim.zero_grad()
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optim.step()
                epoch_loss += (loss.item())
            epoch_time = round(self.getTime() - epoch_time, 4)
            epoch_loss = round(epoch_loss / len(data_loader), 4)
            excel.write(self.current_epoch, 0, self.current_epoch)
            excel.write(self.current_epoch, 1, epoch_loss)
            excel.write(self.current_epoch, 2, self.optim.param_groups[0]['lr'])
            excel.write(self.current_epoch, 3, epoch_time)
            self.lr_scheduler.step()

        # 测试、保存权重
        if (self.test_per_epoch != 0 and self.current_epoch % self.test_per_epoch == 0) or (
                self.current_epoch == self.max_epoch):
            self.test()
            self.logger.save(path=self.log_file_path)

        if (self.save_model_per_epoch != 0 and self.current_epoch % self.save_model_per_epoch == 0) or (
                self.current_epoch == self.max_epoch):
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_root, self.prefix + '_' + str(self.current_epoch) + '.pth')
            )

    def test(self):
        print('[Test:{}/{}]'.format(self.current_epoch, self.max_epoch))
        for i, data_loader in enumerate(self.test_data_loader):
            self.model.eval()
            self.metrics.reset()
            excel = self.logger.work_book.get_sheet(Config.DATASETS_NAME_TEST[i])
            epoch_loss = 0.0
            epoch_time = self.getTime()
            for inputs, masks in data_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, masks)
                    epoch_loss += (loss.item())
                    self.metrics.add_batch(pred=outputs.detach().sigmoid(), mask=masks.detach(), threshold=0.5)
            epoch_time = round(self.getTime() - epoch_time, 4)
            epoch_loss = round(epoch_loss / len(data_loader), 4)
            print(' loss:' + str(epoch_loss) + ' ' + self.metrics.show() + ' ' +data_loader.dataset.dataset_name)
            excel.write(self.current_epoch, 0, self.current_epoch)
            excel.write(self.current_epoch, 1, epoch_loss)
            excel.write(self.current_epoch, 2, round(np.mean(self.metrics.postive_iou), 4))
            excel.write(self.current_epoch, 3, round(np.mean(self.metrics.dice), 4))
            excel.write(self.current_epoch, 4, round(np.mean(self.metrics.f1), 4))
            excel.write(self.current_epoch, 5, round(np.mean(self.metrics.mae), 4))
            excel.write(self.current_epoch, 6, round(np.mean(self.metrics.accuracy), 4))
            excel.write(self.current_epoch, 7, round(np.mean(self.metrics.precision), 4))
            excel.write(self.current_epoch, 8, round(np.mean(self.metrics.recall), 4))
            excel.write(self.current_epoch, 9, epoch_time)

    def getTime(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        else:
            print('!cuda不可用,此时记录的时间会有偏差!')
        return time.time()
