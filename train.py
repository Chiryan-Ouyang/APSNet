import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models import *
from utils import Trainer
import random
import csv
import yaml
import argparse


def default_loader(path, is_img):
    if is_img:
        img = Image.open(path).convert('L')
        img = img.resize((256, 256), Image.LANCZOS)
    else:
        img = Image.open(path).convert('1')
        img = img.resize((64, 64), Image.LANCZOS)
    return img

class MyDataset(Dataset):
    def __init__(self, mode, txt, transform=None, loader=default_loader, data_root=''):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.mode = mode
        self.transform = transform
        self.loader = loader
        self.data_root = data_root

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(self.data_root + 'img/' + fn, True)
        tar = self.loader(self.data_root + 'msk/' + fn, False) 
        
        
        angle = random.uniform(-20, 20)
        img = img.rotate(angle)
        tar = tar.rotate(angle)   
            
        if random.randint(0, 1) and self.mode == 'train':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            tar = tar.transpose(Image.FLIP_LEFT_RIGHT)
        
        img = self.transform(img)
        tar = torch.from_numpy(np.asarray(tar, dtype=np.float32))
        tar = torch.unsqueeze(tar, 0)
        return img, tar, label

    def __len__(self):
        return len(self.imgs)


def main(config):
    batch_size = config['batch_size']
    data_root = config['data_root']
    label_root = config['label_root']
    save_root = config['save_root']
    img_mean = config['img_mean']
    img_std = config['img_std']
    base_lr = config['base_lr']
    max_epoch = config['max_epoch']
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    model_name = config['model']
    block_type = config['block_type']  
    num_heads = config['num_heads'] 
    
    if model_name == 'APSnet18':
        model = APSnet18(use_se=config['use_se'], use_dual_path=config['use_dual_path'], 
                          block_type=config['block_type'], num_heads=config['num_heads'])
    elif model_name == 'APSnet34':
        model = APSnet34(use_se=config['use_se'], use_dual_path=config['use_dual_path'], 
                          block_type=config['block_type'], num_heads=config['num_heads'])
    elif model_name == 'APSnet50':
        model = APSnet50(use_se=config['use_se'], use_dual_path=config['use_dual_path'], 
                          block_type=config['block_type'], num_heads=config['num_heads'])
    else:
        raise ValueError(f"Unsupport Model: {model_name}")

    train_data = MyDataset(
        mode='train',
        txt=label_root + config['train_label_file'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean],
                                 std=[img_std])
        ]),
        data_root=data_root)
    test_data = MyDataset(
        mode='test',
        txt=label_root + config['test_label_file'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[img_mean],
                                 std=[img_std])
        ]),
        data_root=data_root)

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', True),
        drop_last=True)
        
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', True))

    optimizer_name = config.get('optimizer', 'sgd')
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(params=model.parameters(),
                           lr=base_lr, 
                           momentum=config.get('momentum', 0.9), 
                           weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(params=model.parameters(),
                            lr=base_lr, 
                            weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(),
                              lr=base_lr,
                              weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'radam':
        optimizer = optim.RAdam(params=model.parameters(),
                              lr=base_lr,
                              weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'adagrad':
        optimizer = optim.Adagrad(params=model.parameters(),
                                lr=base_lr,
                                weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'adadelta':
        optimizer = optim.Adadelta(params=model.parameters(),
                                 lr=base_lr,
                                 weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(params=model.parameters(),
                                lr=base_lr,
                                momentum=config.get('momentum', 0.9),
                                weight_decay=config.get('weight_decay', 1e-5))
    elif optimizer_name.lower() == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(params=model.parameters(), 
                          lr=base_lr,
                          weight_decay=config.get('weight_decay', 1e-5))
        except ImportError:
            print("Lion optimizer not available, please install with: pip install lion-pytorch")
            print("Falling back to AdamW optimizer")
            optimizer = optim.AdamW(params=model.parameters(), 
                                 lr=base_lr,
                                 weight_decay=config.get('weight_decay', 1e-5))
    else:
        raise ValueError(f"Unsupport Optimizer: {optimizer_name}")

    # 学习率调度器配置
    scheduler_name = config.get('scheduler', 'step')
    if scheduler_name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('lr_step_size', 10), 
            gamma=config.get('lr_gamma', 0.1))
    elif scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epoch)
    else:
        raise ValueError(f"Unsupport scheduler name: {scheduler_name}")
    
    trainer = Trainer(model, optimizer, save_dir=save_root)
    trainer.loop(
        max_epoch, 
        train_loader, 
        test_loader, 
        scheduler, 
        save_freq=config.get('save_freq', 50), 
        test_freq=config.get('test_freq', 10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APSNet Train')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to the yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'cuda_devices' in config:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_devices']
    
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    
    if 'random_seed' in config:
        seed = config['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    main(config)