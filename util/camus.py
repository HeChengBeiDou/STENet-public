r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2
import random
import albumentations
from albumentations import Blur, GaussNoise
from scipy.ndimage import distance_transform_edt as distance

class DatasetCAMUS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size=224, use_original_imgsize=False, trn_image_only_operator=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'#是否训练集
        self.fold = fold
        self.nfolds = 1#无用
        self.nclass = 4
        self.benchmark = 'camus'
        self.shot = shot
        self.img_size = img_size
        # self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'imgs/')
        self.mask_path = os.path.join(datapath, 'masks/')
        self.transform = transform
        self.trn_image_only_operator = trn_image_only_operator

        self.img_file_list, self.mask_file_list = self._prepare_file_list()
        ##确定切面
        img_file_list = []
        mask_file_list = []
        curr_sec = "A2C"
        for i in range(len(self.img_file_list)):
            if curr_sec in self.img_file_list[i] and "MLX" in self.img_file_list[i]:
                img_file_list.append(self.img_file_list[i])
            if curr_sec in self.mask_file_list[i] and "MLX" in self.img_file_list[i]:
                mask_file_list.append(self.mask_file_list[i])
        # self.class_ids = [1, 3]#所有非测试/验证类别
        self.class_ids = [2] if self.split in ['val', 'test'] else [1,3] #query类，在训练阶段是 self.class_ids  测试阶段是其他类
        self.label_value = [0,  63, 127, 191]
        # 测试阶段可以随机跑三回 也可以固定若干个support 放到单独文件夹 但是都需要保证support和query没有

    def get_valid_region(self, s_x):
        s_x = torch.tensor(s_x, dtype=torch.float32)
        kernel = torch.ones(1, 1, 3, 3)
        kernel.requires_grad = False
        kernel = kernel.to(dtype=s_x.dtype, device=s_x.device)

        conv = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False, padding=(1,1))

        conv.weight.data = kernel
        conv.weight.requires_grad = False

        valid_region = (conv(s_x)>0).to(dtype=s_x.dtype)
        return valid_region

    def _prepare_file_list(self):
        img_file_list = os.listdir(self.img_path)
        img_file_list.sort()
        mask_file_list = os.listdir(self.mask_path)
        mask_file_list.sort()
        return img_file_list, mask_file_list

    def __len__(self):
        return len(self.img_file_list)
        # return len(self.img_metadata) if self.split == 'trn' else 1000


    def compute_sdf(self, img_gt):
        """
        img_gt: numpy array (h, w)
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size,1, x, y)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """
        img_gt = img_gt.astype(np.uint8)

        posmask = img_gt.astype(np.uint8)
        if posmask.any():
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)

            contours, _ = cv2.findContours(posmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boundary = np.zeros_like(posmask, np.uint8)
            cv2.drawContours(boundary, contours, -1, (1), 1, 8)

            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
        return sdf


    def __getitem__(self, idx):
        '''
        1. 数据集名称列表不用打乱,因为训练过程有shuffle 先按照idx选择一个数据
        2. 例如,选择心肌为测试,则心肌不出现在训练集;心房和心室数据作为训练集,每次挑一个support和一个query进行训练。
        
        '''
        curr_rand_idx = random.randint(0,len(self.class_ids)-1)
        curr_choose_cls = self.class_ids[curr_rand_idx]
        curr_choose_value = self.label_value[curr_choose_cls]

        ## 选择query图像
        curr_query_idx = idx

        curr_query_img_file_name = self.img_file_list[curr_query_idx]
        curr_query_mask_file_name = self.mask_file_list[curr_query_idx]

        tmp = curr_query_img_file_name.split('.')[0].split('_')
        curr_query_patient = tmp[0]
        curr_query_view = tmp[1]
        curr_query_time = tmp[2]

        curr_query_img = cv2.imread(os.path.join(self.img_path, curr_query_img_file_name), flags=cv2.IMREAD_GRAYSCALE)
        curr_query_mask = cv2.imread(os.path.join(self.mask_path, curr_query_mask_file_name), flags=cv2.IMREAD_GRAYSCALE)

        query_pix = np.where(curr_query_mask==curr_choose_cls)
        curr_query_choose_mask = (curr_query_mask > 0).astype(np.uint8)

        # support与query相同方法增强
        if self.transform:
            augmented = self.transform(image=curr_query_img, mask=curr_query_choose_mask)
            curr_query_img = augmented['image']
            curr_query_choose_mask = augmented['mask']
            fix_apply = augmented['replay']

        curr_query_choose_sdm =  self.compute_sdf(curr_query_choose_mask)

        ## 找到k个shot
        support_imgs = []
        support_masks = []
        # support_idxs = []
        support_patients = []
        support_names = []

        rand_ = np.random.randint(1, len(self.img_file_list))
        curr_idx = (curr_query_idx + rand_)%len(self.img_file_list)# TODO  这里不能只加1

        # random.randint(0,len(self.img_file_list)-1)
        while len(support_imgs) < self.shot:
            # print(curr_idx)
            curr_support_img_file_name = self.img_file_list[curr_idx]
            curr_support_mask_file_name = self.mask_file_list[curr_idx]

            tmp_file_name,file_type = curr_support_img_file_name.split('.')
            patient, section, temporal = tmp_file_name.split('_')
            rand_ = np.random.randint(1, len(self.img_file_list))
            curr_idx = (curr_idx+rand_)%len(self.img_file_list)
            if curr_query_patient not in support_patients and patient != curr_query_patient and curr_query_view == section:
                support_patients.append(patient)
                curr_support_img = cv2.imread(os.path.join(self.img_path, curr_support_img_file_name), flags=cv2.IMREAD_GRAYSCALE)
                curr_support_mask = cv2.imread(os.path.join(self.mask_path, curr_support_mask_file_name), flags=cv2.IMREAD_GRAYSCALE)

                support_names.append(curr_support_img_file_name.split('.')[0])

                # curr_support_mask_label = np.unique(curr_support_mask)

                # assert len(curr_support_mask_label) == 4
                curr_support_choose_mask = (curr_support_mask>0).astype(np.uint8)#当前选择的support类mask
                
                if np.sum(curr_support_choose_mask) == 0:
                    print(curr_idx)
                    continue
                ## k shot 1 way  所以类别都是一个
                support_imgs.append(curr_support_img)
                support_masks.append(curr_support_choose_mask)



        ## 数据增强 albumentations
        if self.transform is not None:
            tmp_support_imgs = []
            tmp_support_masks = []
            tmp_support_sdm = []
            for tmp_i in range(len(support_imgs)):
                augmented = albumentations.ReplayCompose.replay(saved_augmentations=fix_apply,image=support_imgs[tmp_i], mask=support_masks[tmp_i],)
                curr_support_img = augmented['image']
                curr_support_choose_mask = augmented['mask']

                ## 增加SDM
                curr_support_choose_sdm = self.compute_sdf(curr_support_choose_mask)
                

                curr_s_y_valid_region = self.get_valid_region(curr_support_img[None,None,...])
                curr_s_y_valid_region = curr_s_y_valid_region.numpy()[0,0,...]
                if self.trn_image_only_operator is not None:
                    trn_image_only_augmented = self.trn_image_only_operator(image=curr_support_img)
                    curr_support_img = trn_image_only_augmented['image']

                #curr_support_img = curr_support_img * curr_s_y_valid_region

                tmp_support_imgs.append(curr_support_img)
                tmp_support_masks.append(curr_support_choose_mask)
                tmp_support_sdm.append(curr_support_choose_sdm)
            
            support_imgs = np.array(tmp_support_imgs)
            support_masks = np.array(tmp_support_masks)
            support_sdms = np.array(tmp_support_sdm)

        curr_query_img = torch.as_tensor(curr_query_img).float().contiguous()
        curr_query_img.unsqueeze_(dim=0)
        # curr_query_img = curr_query_img.permute(2, 0, 1)
        curr_query_choose_mask = torch.as_tensor(curr_query_choose_mask).float().contiguous()
        # curr_query_choose_mask = curr_query_choose_mask.permute(2, 0, 1)
        curr_query_choose_mask.unsqueeze_(dim=0)
        curr_query_choose_sdm = torch.as_tensor(curr_query_choose_sdm).float().contiguous()

        support_imgs = torch.as_tensor(support_imgs).float().contiguous()
        support_imgs.unsqueeze_(dim=1)
        # support_imgs = support_imgs.permute(0,3,1,2)
        support_masks = torch.as_tensor(support_masks).float().contiguous()
        # support_masks = support_masks.permute(0,3,1,2)
        support_masks.unsqueeze_(dim=1)
        support_sdms = torch.as_tensor(support_sdms).float().contiguous()

        curr_query_img /= 255
        support_imgs /= 255

        batch = {'query_img': torch.as_tensor(curr_query_img),#(1,h,w)
                 'query_mask': torch.as_tensor(curr_query_choose_mask),#(h,w)
                 'query_sdm': torch.as_tensor(curr_query_choose_sdm),

                 'support_imgs': torch.as_tensor(support_imgs),#(shot,1,h,w)
                 'support_masks': torch.as_tensor(support_masks),#(shot,1,h,w)
                 'support_sdms': torch.as_tensor(support_sdms),#(shot,1,h,w)

                 'class_id': torch.as_tensor(curr_choose_cls),
                 'support_names':support_names,
                 'query_name':curr_query_img_file_name.split('.')[0]
                 }
        return batch


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append(r"../../")
    import config
    from camus import DatasetCAMUS
    from albumentations import RandomRotate90,Resize,GaussianBlur,HorizontalFlip,Affine,Rotate,ShiftScaleRotate,ReplayCompose,Flip

    def get_configs():
        parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
        # parser.add_argument('--arch', type=str, default='BAM') # 
        # parser.add_argument('--viz', action='store_true', default=False)
        parser.add_argument('--config', type=str, default='../config/camus/camus_settings.yaml', help='config file') # coco/coco_split0_resnet50.yaml
        parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
        parser.add_argument('--opts', help='see config/camus/camus.yaml for all options', default=None, nargs=argparse.REMAINDER)#在脚本中定义了一个命令行参数 --opts，允许用户通过命令行输入一系列额外的参数，这些参数将被保存在一个列表
        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)
        cfg = config.merge_cfg_from_args(cfg, args)
        if args.opts is not None:
            cfg = config.merge_cfg_from_list(cfg, args.opts)
        return cfg

    args = get_configs()#args没用到 其实也不需要有
    # print(cfgs.DATA)
    net_params= args.NETWORK

    train_transform = [
        RandomRotate90(),
        Flip(),
        Resize(args["COMMON"].img_size, args["COMMON"].img_size, always_apply=True),
        ]
    train_transform = ReplayCompose(train_transform)

    train_image_only_transform = ReplayCompose([
        GaussNoise(), 
        Blur(),]
        )

    val_transform = [
        Resize(args["COMMON"].img_size, args["COMMON"].img_size, always_apply=True),
    ]
    val_transform = ReplayCompose(val_transform)

    batch_size = 2
    shot = 1
    img_size = 224


    dataset = DatasetCAMUS(datapath=args["TRAIN"].datapath,fold=args["COMMON"].fold, transform=train_transform, split='trn', shot=args["NETWORK"].shot,trn_image_only_operator=train_image_only_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, \
                                                pin_memory=True, drop_last=False, \
                                                shuffle=False)
    for idx, batch in enumerate(tqdm(dataloader)):#tqdm(dataloader)
        query_img = batch['query_img']#(n,c,h,w)
        query_mask = batch['query_mask'].clone()#(n,1,h,w)
        support_imgs = batch['support_imgs']#(n,shot,c,h,w)
        support_masks = batch['support_masks'].clone()#(n,shot,1,h,w)

        batch['query_mask'].squeeze_(dim=1)#(n,h,w)

        print(query_img.shape)

