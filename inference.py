import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import Config
import PolypDatasets
from Utils.loss import structure_loss, structure_loss_PraNet,structure_loss_x3
from Utils.metrics import metrics
def getTime():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        print('!cuda不可用,此时记录的时间会有偏差!')
    return time.perf_counter()


if __name__ == '__main__':
    pth_path = r''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from Model.LPDNet import LPDNet
    model = LPDNet().to(device)
    # model.load_state_dict(torch.load(pth_path))  # GPU
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))  # CPU
    model.eval()
    criterion = structure_loss()
    metrics = metrics()
    test_data_loader = PolypDatasets.get_test_data_loader(BATCH_SIZE=1)
    warming_input = torch.randn([10, 3, 352, 352]).to(device)
    warming_out = model(warming_input)

    for i, data_loader in enumerate(test_data_loader):

        metrics.reset()
        epoch_loss = 0.0
        epoch_time = 0.0
        for inputs, masks in data_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                # ----------------------
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.perf_counter() - inference_time
                # ----------------------
                loss = criterion(outputs, masks)
                epoch_loss += (loss.item())
                epoch_time += inference_time
                metrics.add_batch(pred=outputs.detach().sigmoid(), mask=masks.detach(), threshold=0.5)
        print('--- --- ---')
        print('DatasetName:{} Dataset Length:{} Total Inference Time:{} Avg Inference Time:{} FPS:{}'.format(
            data_loader.dataset.dataset_name, len(data_loader.dataset), epoch_time, epoch_time / len(data_loader.dataset), 1 / (epoch_time / len(data_loader.dataset))))
        print(metrics.show())

