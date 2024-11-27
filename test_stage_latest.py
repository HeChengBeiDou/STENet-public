import torch
from torch.utils.data import DataLoader
import timm
from datasets.fetalUS_dataset import BaseDataSets, RandomGenerator
from tensorboardX import SummaryWriter
# from models.vmunet.vmunet import VMUNet
from models.unet.unet import UNet

from engine import *
import os
import sys

from util.utils import *
from configs.config_setting_fetalUS import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    config.work_dir = r"./results/vmunet_isic18_Wednesday_09_October_2024_18h_37m_49s/"
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)
    # global writer
    # writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    # torch.cuda.empty_cache()
    test_transform = config.test_transformer

    print('#----------Preparing dataset----------#')
    config.stage = "test"
    test_dataset = BaseDataSets(
        base_dir=config.base_dir,
        split="test",
        num=None,
        transform=test_transform,
        ops_weak=None,
        ops_strong=None)
    # NPY_datasets(config.data_path, config, train=False)
    test_loader = DataLoader(test_dataset,
                                batch_size=10,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    # if config.network == 'vmunet':
    #     model = VMUNet(
    #         num_classes=model_cfg['num_classes'],
    #         input_channels=model_cfg['input_channels'],
    #         depths=model_cfg['depths'],
    #         depths_decoder=model_cfg['depths_decoder'],
    #         drop_path_rate=model_cfg['drop_path_rate'],
    #         load_ckpt_path=model_cfg['load_ckpt_path'],
    #     )
    #     # model.load_from()
        
    # else: raise Exception('network in not right!')
    model = UNet(n_channels=model_cfg['input_channels'], n_classes=config.num_classes, bilinear=True)
    # cal_params_flops(model, 256, logger)

    device = torch.device("cuda:{}".format(
        config.device_ids[0]) if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    
    # model.to(device=device)
    # cal_params_flops(model, 256, logger,device=device)

    
    model = nn.DataParallel(model, device_ids=config.device_ids)
    model.to(device=device)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion


    print('#----------Set other params----------#')
    print('#----------Training----------#')

    # latest_weight = torch.load('./results/vmunet_isic18_Tuesday_10_September_2024_10h_41m_35s/checkpoints/latest.pth')['model_state_dict']
    # model.load_state_dict(latest_weight)
    # loss = test_one_epoch(
    #         test_loader,
    #         model,
    #         criterion,
    #         logger,
    #         config,
    #         device=device
            
    #     )
    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/latest.pth', map_location=device)['model_state_dict']
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
                device=device
                
            )




    #     # os.rename(
    #     #     os.path.join(checkpoint_dir, 'best.pth'),
    #     #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    #     # )      

    # loss = test_one_epoch(
    #         test_loader,
    #         model,
    #         criterion,
    #         logger,
    #         config,
    #         device=device
    #     )

if __name__ == '__main__':
    config = setting_config
    main(config)