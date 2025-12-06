from scipy.ndimage.interpolation import zoom
import os
import re
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra

from data.Image_loader import CoronaryImage
from model.FCN import FCN_Gate, FCN
from model.loss import DiceLoss, DiceLoss_v1
from utils.Calculate_metrics import Cal_metrics
from utils.parallel import parallel
from utils.utils import get_csv_split, reshape_img


def train(model, criterion, train_loader, opt, device, e):
    model = model.to(device)
    model.train()
    train_sum = 0
    for j, batch in enumerate(train_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)
        outputs = model(img)
        opt.zero_grad()
        loss = criterion(outputs, label)
        print('Epoch {:<3d}  |  Step {:>3d}/{:<3d}  | train loss {:.4f}'.format(e, j, len(train_loader), loss.item()))
        train_sum += loss.item()
        loss.backward()
        opt.step()
    return train_sum / len(train_loader)


def valid(model, criterion, valid_loader, device, e):
    model.eval()
    valid_sum = 0
    for j, batch in enumerate(valid_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, label)
        valid_sum += loss.item()
        print('Epoch {:<3d}  |Step {:>3d}/{:<3d}  | valid loss {:.4f}'.format(e, j, len(valid_loader), loss.item()))

    return valid_sum / len(valid_loader)


def inference(model, criterion, train_loader, valid_loader, device, save_img_path, is_infer_train=True):
    model.eval()
    if is_infer_train:
        for batch in tqdm(train_loader):
            img, label = batch['image'].float(), batch['label'].float()
            img, label = img.to(device), label.to(device)
            # file_name = batch['id_index'][0]

            with torch.no_grad():
                outputs = model(img)
                loss = criterion(outputs, label)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)
            pre = outputs.cpu().detach().numpy()

            ID = batch['image_index']
            affine = batch['affine']
            img_size = batch['image_size']
            os.makedirs(save_img_path, exist_ok=True)
            batch_save(ID, affine, pre, img_size, save_img_path)

    for batch in tqdm(valid_loader):
        img, label = batch['image'].float(), batch['label'].float()
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(img)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze(1)
        pre = outputs.cpu().detach().numpy()

        ID = batch['image_index']
        affine = batch['affine']
        img_size = batch['image_size']
        os.makedirs(save_img_path, exist_ok=True)
        batch_save(ID, affine, pre, img_size, save_img_path)


def batch_save(ID, affine, pre, img_size, save_img_path):
    batch_size = len(ID)
    save_list = [save_img_path] * batch_size
    parallel(save_picture, pre, affine, img_size, save_list, ID, thread=True)


def save_picture(pre, affine, img_size, save_name, id):
    pre_label = pre
    pre_label[pre_label >= 0.5] = 1
    pre_label[pre_label < 0.5] = 0
    pre_label = reshape_img(pre_label, img_size.numpy())
    os.makedirs(os.path.join(save_name, id), exist_ok=True)
    nib.save(nib.Nifti1Image(pre_label, affine), os.path.join(save_name, id + '/pre_label.nii.gz'))

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for direct segmentation training / inference.

    Usage examples:
      - python direct_seg.py
      - python direct_seg.py train.fold=2 train.model=FCN
      - python direct_seg.py general.csv_path=data_list/split_v2.csv
    """
    general_cfg = cfg.general
    train_cfg = cfg.train

    # Paths and global settings (resolve relative paths against original working dir)
    learning_rate = float(general_cfg.lr)
    train_path = to_absolute_path(str(general_cfg.data_path))
    valid_path = to_absolute_path(str(general_cfg.data_path))
    csv_path = to_absolute_path(str(general_cfg.csv_path))

    # Training / runtime parameters
    gpu_index = int(train_cfg.gpu_index)
    k = int(train_cfg.fold)
    load_num = int(train_cfg.load_num)
    batch_size = int(train_cfg.batch_size)
    model_name = str(train_cfg.model)
    channel = int(train_cfg.channel)
    resolution = int(train_cfg.rl)
    pool_nums = int(train_cfg.pools)
    num_workers = int(train_cfg.num_workers)
    is_train = int(train_cfg.is_train)
    loss_name = str(train_cfg.loss)
    epochs = int(train_cfg.epochs)
    is_inference_flag = int(train_cfg.is_inference)

    result_path = r'result/Direct_seg'

    # Device setup: use CUDA if available, otherwise fall back to CPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training.")

    if resolution == 1:
        resolution_name = 'High_resolution'
        input_size = [512, 512, 256]
    elif resolution == 2:
        resolution_name = 'Mid_resolution'
        input_size = [256, 256, 128]
    elif resolution == 3:
        resolution_name = 'Low_resolution'
        input_size = [128, 128, 64]
    else:
        raise ValueError("没有该级别的分辨率")

    parameter_record = resolution_name + '_%d_' % channel + loss_name

    print('model: %s || parameters: %s || %d_fold' % (model_name, parameter_record, k))

    # 读取参数配置文件
    model_save_path = r'%s/%s/%s/fold_%d/model_save' % (result_path, model_name, parameter_record, k)
    save_label_path = r'%s/%s/%s/fold_%d/pre_label' % (result_path, model_name, parameter_record, k)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    ID_list = get_csv_split(csv_path, k)

    # 数据加载
    train_set = CoronaryImage(train_path, train_path, ID_list['train'], input_size)
    valid_set = CoronaryImage(valid_path, valid_path, ID_list['valid'], input_size)
    train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers, shuffle=False)

    # 网络模型
    if model_name == "FCN":
        net = FCN(channel).to(device)
    elif model_name == "FCN_AG":
        net = FCN_Gate(channel).to(device)
    else:
        raise ValueError("模型错误")

    model_list = os.listdir(model_save_path)

    # 是否加载网络模型
    if load_num == 0:
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
    else:
        net.load_state_dict(torch.load(model_save_path + '/net_%d.pkl' % load_num, map_location="cpu"))
        load_num = load_num + 1

    # Finally move the network to the selected device (CUDA / CPU)
    net = net.to(device)

    net_opt = optim.Adam(net.parameters(), lr=learning_rate)
    if loss_name == 'Dice':
        criterion = DiceLoss()
    elif loss_name == 'Dice_dilation':
        criterion = DiceLoss_v1()
    else:
        raise ValueError("No loss")

    train_loss_set = []
    valid_loss_set = []
    epoch_list = []

    # 训练
    if is_train == 1:
        for e in range(load_num, epochs):
            print("=============train=============")
            train_loss = train(net, criterion, train_loader, net_opt, device, e)
            print("=============valid=============")
            valid_loss = valid(net, criterion, valid_loader, device, e)
            # valid_loss = 0
            train_loss_set.append(train_loss)
            valid_loss_set.append(valid_loss)
            epoch_list.append(e)
            print("train_loss:%f || valid_loss:%f" % (train_loss, valid_loss))
            if e % 5 == 0:
                torch.save(net.state_dict(), model_save_path + '/net_%d.pkl' % e)
        record = dict()
        record['epoch'] = epoch_list
        record['train_loss'] = train_loss_set
        record['valid_loss'] = valid_loss_set
        record = pd.DataFrame(record)
        record_name = time.strftime("%Y_%m_%d_%H.csv", time.localtime())
        record.to_csv(r'%s/%s/%s/fold_%d/%s' % (result_path, model_name, parameter_record, k, record_name),
                      index=False)

    # 推断
    train_infer_loader = DataLoader(train_set, 2, num_workers=num_workers, shuffle=False)
    valid_infer_loader = DataLoader(valid_set, 2, num_workers=num_workers, shuffle=False)

    if is_inference_flag == 1:
        print('now inference..............')
        inference(net, criterion, train_infer_loader, valid_infer_loader, device, save_label_path,
                  is_infer_train=True)

    # 计算最后的dice
    print('now calculate dice...........')
    CD = Cal_metrics(os.path.join(save_label_path), valid_path, 0)
    p = multiprocessing.Pool(pool_nums)
    result = p.map(CD.calculate_dice, ID_list['valid'])
    p.close()
    p.join()

    # 保存结果
    record_dice = {}
    record_dice['ID'] = ID_list['valid']
    result = np.array(result)
    record_dice['dice'] = result[:, 0]
    record_dice['ahd'] = result[:, 1]
    record_dice['hd'] = result[:, 2]
    record_dice = pd.DataFrame(record_dice)
    record_dice.to_csv(r'%s/%s/%s/fold_%d/result.csv' % (result_path, model_name, parameter_record, k), index=False)


if __name__ == '__main__':
    main()
