import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import model_end_end
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from emcadnet import EMCADNet
from albumentations import RandomRotate90, Resize
import archs

import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import time
import sys 
sys.path.append('/mnt/Rolling-Unet-free-isic/')
# from xr_model27_7_1 import Rolling_Unet_S 
# from networks.multiresunet import Rolling_Unet_S
# from networks.focalunetr import Rolling_Unet_S
# from networks.hdenseunet import Rolling_Unet_S
# from networks.hdenseformer_hatb import Rolling_Unet_S_HATB as Rolling_Unet_S
# from networks.MEW_UNet import Rolling_Unet_S
# from mixedunet import Rolling_Unet_S 
# from mewunet import Rolling_Unet_S
# from networks.bra_unet import Rolling_Unet_S 
# from networks.perspective_unet import Rolling_Unet_S
# from networks.HVNet import Rolling_Unet_S
# from networks.mambaunet import Rolling_Unet_S
# from networks.Missformer import Rolling_Unet_S
# from contrast_models.nnunetv2_swin_umamba.nets.SwinUMamba import Rolling_Unet_S
# from networks.swinunet import Rolling_Unet_S
# from networks.MERIT import Rolling_Unet_S
# from contrast_models.CASCADE_lib.networks import Rolling_Unet_S
# from contrast_models.transunet_networks.vit_seg_modeling import Rolling_Unet_S
# from deformabledlka_2d import Rolling_Unet_S
# from att_unet import Rolling_Unet_S
# from unetr import Rolling_Unet_S 
from xr_model27_7_3 import Rolling_Unet_S 
# from networks.Hiformer import Rolling_Unet_S
# from networks.utnet import Rolling_Unet_S
# from networks.ConvUNeXt import Rolling_Unet_S

from tensorboardX import SummaryWriter
ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


"""
需要跑通一个数据集注意：
数据集的命名格式，需要参照dataset.py文件中60行
注意需要将数据裁剪到哪个尺寸
"""
"""
2024-5-27.16.13 没有改变模型任何地方-数据集处理时报错：检查图像的宽高不一致
在compse函数中禁用检查 

2024-5-27.19.28-isic-改变训练、验证集划分比例 6:4-epoch-200-batchsize-8
2024-5-28.11.07-glas训练、验证集划分比例 8:2 -epoch-400-batchsize-8-inputsize-512
2024-5-28.20.05-busi训练、验证集划分比例 8:2 -epoch-400-batchsize-8-inputaize-256
2024-5-28.17.18-glas训练、验证划分比列 8:2 -epoch-400-bs-8-512-在double transformer基础上，在lo2内部加入
多尺度特征图--efficientvit-litLMA模块中的
"""
"""
对于chasedb1数据集lr为0.001，weight_decay为0.001
"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=6, type=int, metavar='N', help='mini-batch size(default: 8)')
    parser.add_argument('--num_workers', default=4, type=int)

    # model
    parser.add_argument('--train_model_name', '-t', 
                        default='ori_model10')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='Rolling_Unet_S')  ### Rolling_Unet_S, Rolling_Unet_M, Rolling_Unet_L
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width(default: 256)')
    parser.add_argument('--input_h', default=256, type=int, help='image height(default: 256)')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # data
    parser.add_argument('--dataset', default='isic', help='dataset name')  ### isic, busi, chasedb1, glas
    parser.add_argument('--img_ext', default='.png', help='image file extension')
    parser.add_argument('--mask_ext', default='.png', help='masks file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR',
                        help='initial learning rate(default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')
    parser.add_argument('--resume', default=False, type=str2bool, help='nesterov')
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
      
        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            # print(f'output-class:{type(output)}')
            # print(f'train-output.shape:{output.shape}')
            loss = criterion(output, target)
            iou, dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_loss_curves(name, train_loss, val_loss):
    
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.savefig('models/%s/loss_curve.png' % name)
    
    plt.close()



def read_img_ids_from_file(file_name):  
    """  
    从指定的txt文件中读取图片ID，并返回一个列表。  
    
    :param file_name: 要读取的txt文件名  
    :return: 包含图片ID的列表  
    """  
    img_ids = []  # 创建一个空列表来存储读取的图片ID  

    # 打开文件进行读取  
    with open(file_name, 'r') as file:  
        for line in file:  
            img_ids.append(line.strip())  # 使用 strip() 去除每行的换行符  

    return img_ids 
    
def main():
    seed_torch()
    config = vars(parse_args())

    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    my_writer = SummaryWriter()

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()
        
    cudnn.benchmark = True
    
    # create model
   
    
    model =Rolling_Unet_S(num_classes=config['num_classes'],
                       input_channels= config['input_channels'],
                    deep_supervision=config['deep_supervision'],
                    img_size=config['input_w'])
    
    # model = EMCADNet(num_classes=config['num_classes'], kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, 
    #                  add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir=r'predtained_pth')
    # model =deformabledlka_2d.__dict__[config['arch']](num_classes=config['num_classes'],
    #                                        input_channels=config['input_channels'],
    #                                        deep_supervision=config['deep_supervision'],
    #                                        img_size=config['input_w'])
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError



    # train_img_ids=read_img_ids_from_file(r'inputs\parnet_data_Kvasir_04\train_list.txt')
    # val_img_ids=read_img_ids_from_file(r'inputs\parnet_data_Kvasir_04\test_list.txt')

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
   
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    ###每个数据集进行三次实验：random_state分别为41,1029,3407 
    ###其中chasedb1数据集test_size为0.27这样训练图像个数为20张即为bs的倍数
    print(f'len_train_img_ids:{len(train_img_ids)},val_img_ids:{len(val_img_ids)}')
    """
    这里在划分训练集与测试集时绝对不可以将图像按照顺序划分
    """
    print(f'----------------------------------------------------------')
    print(f'train_img_ids:{train_img_ids}')
    print(f'\nval_img_ids:{val_img_ids}')
    train_transform = Compose([
        RandomRotate90(),
        albumentations.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ],is_check_shapes=False)
   
   
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ],is_check_shapes=False,)

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        # img_dir= r'inputs\parnet_data_Kvasir_04\images',
        # mask_dir=r'inputs\parnet_data_Kvasir_04\masks',
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        # img_dir= r'inputs\parnet_data_Kvasir_04\images',
        # mask_dir=r'inputs\parnet_data_Kvasir_04\masks',
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0

    #存放训练，验证损失列表
    train_loss = []
    val_loss = []

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        train_loss.append(train_log['loss'])
        val_loss.append(val_log['loss'])
        print(f"train_log-loss:{train_log['loss']},val_loss:{val_log['loss']}")
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        my_writer.add_scalar('loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val_iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_dice', val_log['dice'], global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        
        

        torch.cuda.empty_cache()
        # if epoch % 5== 0:  # 每10个epoch绘制一次
        plot_loss_curves(config['name'], train_loss, val_loss)
       

    plot_loss_curves(config['name'],train_loss, val_loss)

if __name__ == '__main__':
    main()
