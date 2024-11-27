import torch
from torch.utils.data import DataLoader
import timm
from datasets.fetalUS_dataset import BaseDataSets
from tensorboardX import SummaryWriter
# from models.vmunet.vmunet import VMUNet
from models.unet.unet import UNet
from engine import *
import os
import sys
from torchvision import transforms
from datasets.fetalUS_dataset import BaseDataSets, RandomGenerator

from util.utils import *
from configs.config_setting_fetalUS import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)

    config.stage = "train"


    train_transform = config.train_transformer
    val_transform = config.train_transformer
    print('#----------Preparing dataset----------#')

    train_dataset = BaseDataSets(
        base_dir=config.base_dir,
        split="train",
        num=None,
        transform=train_transform,
        ops_weak=None,
        ops_strong=None)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
    val_dataset = BaseDataSets(
        base_dir=config.base_dir,
        split="val",
        num=None,
        transform=val_transform,
        ops_weak=None,
        ops_strong=None)
    val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    # if config.network == 'vmunet':
    #     model = VMUNet(
    #         num_classes=config.num_classes,
    #         input_channels=model_cfg['input_channels'],
    #         depths=model_cfg['depths'],
    #         depths_decoder=model_cfg['depths_decoder'],
    #         drop_path_rate=model_cfg['drop_path_rate'],
    #         load_ckpt_path=model_cfg['load_ckpt_path'],
    #     )
        
    # else: 
    #     raise Exception('network in not right!')
    model = UNet(n_channels=model_cfg['input_channels'], n_classes=config.num_classes, bilinear=True)
    

    
    device = torch.device("cuda:{}".format(
        config.device_ids[0]) if torch.cuda.is_available() else "cpu")
    
    # model.to(device=device)
    # cal_params_flops(model, 256, logger,device=device)

    
    model = nn.DataParallel(model, device_ids=config.device_ids)
    model.to(device=device)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    min_loss = 0
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)


    step = 0
    print('#----------Training----------#')
    
    for epoch in range(start_epoch, config.epochs + 1):

        # torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                writer,
                device=device
            )

        # if loss < min_loss:
        if loss > min_loss:
            log_info = f'best epoch: {epoch}'
            logger.info(log_info)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    #     print('#----------Testing----------#')
    #     best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=device)
    #     model.load_state_dict(best_weight)
    #     loss = test_one_epoch(
    #             val_loader,
    #             model,
    #             criterion,
    #             logger,
    #             config,
    #         )
    #     os.rename(
    #         os.path.join(checkpoint_dir, 'best.pth'),
    #         os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    #     )      


if __name__ == '__main__':
    config = setting_config
    main(config)