import argparse
import os
from glob import glob
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import sys 
sys.path.append('/mnt/Rolling-Unet-free-isic/')
# from networks.utnet import Rolling_Unet_S
# from networks.hdenseformer import Rolling_Unet_S
# from contrast_models.transunet_networks.vit_seg_modeling import Rolling_Unet_S
# from networks.focalunetr import Rolling_Unet_S
# from networks.multiresunet import Rolling_Unet_S
# # from mixedunet import Rolling_Unet_S 
# from att_unet import Rolling_Unet_S
# from networks.hdenseunet import Rolling_Unet_S
# from xr_model27_7 import Rolling_Unet_S 
from xr_model27_7_3 import Rolling_Unet_S
# from networks.UNeXt import Rolling_Unet_S
# from networks.resunet import Rolling_Unet_S
# from networks.Rollingunet  import Rolling_Unet_Lz
# from  networks.Hiformer import Rolling_Unet_S
# from networks.utnet import Rolling_Unet_S
# from mewunet import Rolling_Unet_S
# # from networks.utnet import Rolling_Unet_S
# from networks.Hiformer import Rolling_Unet_S
# from unetr import Rolling_Unet_S 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='isic_Rolling_Unet_S_woDS', help='model name')
    args = parser.parse_args()
    return args


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    args = parse_args()

    with open(r'%s\config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])

    model =Rolling_Unet_S(num_classes=config['num_classes'],
                       input_channels= config['input_channels'],
                    deep_supervision=config['deep_supervision'],
                    img_size=config['input_w'])

    # model =deformabledlka_2d.__dict__[config['arch']](num_classes=config['num_classes'],
    #                                        input_channels=config['input_channels'],
    #                                        deep_supervision=config['deep_supervision'],
    #                                        img_size=config['input_w'])
    model = model.cuda()

    # # Data loading code
    # val_img_ids=read_img_ids_from_file(r'inputs\parnet_data_Kvasir_04\test_list.txt')

    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    print(f'val_img_ids:{len(val_img_ids)}')
    print(f'\nval_img_ids:{val_img_ids}')
    model.load_state_dict(torch.load('%s\model.pth' % args.name))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

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
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', args.name, str(c)),
                    exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            # iou, dice = iou_score(output, target)
            iou, dice, hd, hd95, recall, specificity, precision = indicators(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd_avg_meter.update(hd, input.size(0))
            hd95_avg_meter.update(hd95, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            precision_avg_meter.update(precision, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Hd: %.4f' % hd_avg_meter.avg)
    print('Hd95: %.4f' % hd95_avg_meter.avg)
    print('Recall: %.4f' % recall_avg_meter.avg)
    print('Specificity: %.4f' % specificity_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()