r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2
import random


class DatasetACDC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size=224, use_original_imgsize=False):
        self.split = 'val' if split in ['val', 'test'] else 'trn'#是否训练集
        self.fold = fold
        self.nfolds = 1#无用
        self.nclass = 4
        self.benchmark = 'acdc'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'imgs/')
        self.mask_path = os.path.join(datapath, 'masks/')
        self.transform = transform

        self.img_file_list, self.mask_file_list = self._prepare_file_list()
        # self.class_ids = [1, 3]#所有非测试/验证类别
        self.class_ids = [2] if self.split in ['val', 'test'] else [1,3] #query类，在训练阶段是 self.class_ids  测试阶段是其他类
        self.label_value = [0,  63, 127, 191]
        # 测试阶段可以随机跑三回 也可以固定若干个support 放到单独文件夹 但是都需要保证support和query没有
    
    def _prepare_file_list(self):
        img_file_list = os.listdir(self.img_path)
        img_file_list.sort()
        mask_file_list = os.listdir(self.mask_path)
        mask_file_list.sort()
        return img_file_list, mask_file_list
    def __len__(self):
        return len(self.img_file_list)
        # return len(self.img_metadata) if self.split == 'trn' else 1000

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

        curr_query_patient = curr_query_img_file_name.split('.')[0].split('_')[0]

        curr_query_img = cv2.imread(os.path.join(self.img_path, curr_query_img_file_name), flags=cv2.IMREAD_GRAYSCALE)
        curr_query_mask = cv2.imread(os.path.join(self.mask_path, curr_query_mask_file_name), flags=cv2.IMREAD_GRAYSCALE)

        query_pix = np.where(curr_query_mask==curr_choose_cls)
        curr_query_choose_mask = (curr_query_mask==curr_choose_value).astype(np.uint8)

        augmented = self.transform(image=curr_query_img, mask=curr_query_choose_mask)
        curr_query_img = augmented['image']
        curr_query_choose_mask = augmented['mask']

        ## 找到k个shot
        support_imgs = []
        support_masks = []
        # support_idxs = []
        support_patients = []
        support_names = []

        while len(support_imgs) < self.shot:
            curr_idx = random.randint(0,len(self.img_file_list)-1)
            curr_support_img_file_name = self.img_file_list[curr_idx]
            curr_support_mask_file_name = self.mask_file_list[curr_idx]

            tmp_file_name,file_type = curr_support_img_file_name.split('.')
            patient, section, temporal = tmp_file_name.split('_')
            if curr_query_patient not in support_patients and patient != curr_query_patient:
                support_patients.append(patient)
                curr_support_img = cv2.imread(os.path.join(self.img_path, curr_support_img_file_name), flags=cv2.IMREAD_GRAYSCALE)
                curr_support_mask = cv2.imread(os.path.join(self.mask_path, curr_support_mask_file_name), flags=cv2.IMREAD_GRAYSCALE)

                support_names.append(curr_support_img_file_name.split('.')[0])

                curr_support_mask_label = np.unique(curr_support_mask)

                assert len(curr_support_mask_label) == 4
                curr_support_choose_mask = (curr_support_mask==curr_choose_value).astype(np.uint8)#当前选择的support类mask

                ## k shot 1 way  所以类别都是一个
                support_imgs.append(curr_support_img)
                support_masks.append(curr_support_choose_mask)



        ## 数据增强 albumentations
        if self.transform is not None:
            tmp_support_imgs = []
            tmp_support_masks = []
            random.seed(random.randint(0,1000))
            for tmp_i in range(len(support_imgs)):
                augmented = self.transform(image=support_imgs[tmp_i], mask=support_masks[tmp_i])
                curr_support_img = augmented['image']
                curr_support_choose_mask = augmented['mask']
                tmp_support_imgs.append(curr_support_img)
                tmp_support_masks.append(curr_support_choose_mask)
            
            support_imgs = np.array(tmp_support_imgs)
            support_masks = np.array(tmp_support_masks)


        curr_query_img = torch.as_tensor(curr_query_img).float().contiguous()
        curr_query_img.unsqueeze_(dim=0)
        # curr_query_img = curr_query_img.permute(2, 0, 1)
        curr_query_choose_mask = torch.as_tensor(curr_query_choose_mask).float().contiguous()
        # curr_query_choose_mask = curr_query_choose_mask.permute(2, 0, 1)
        curr_query_choose_mask.unsqueeze_(dim=0)

        support_imgs = torch.as_tensor(support_imgs).float().contiguous()
        support_imgs.unsqueeze_(dim=1)
        # support_imgs = support_imgs.permute(0,3,1,2)
        support_masks = torch.as_tensor(support_masks).float().contiguous()
        # support_masks = support_masks.permute(0,3,1,2)
        support_masks.unsqueeze_(dim=1)

        curr_query_img /= 255
        support_imgs /= 255

        batch = {'query_img': torch.as_tensor(curr_query_img),#(1,h,w)
                 'query_mask': torch.as_tensor(curr_query_choose_mask),#(h,w)

                 'support_imgs': torch.as_tensor(support_imgs),#(shot,1,h,w)
                 'support_masks': torch.as_tensor(support_masks),#(shot,1,h,w)

                 'class_id': torch.as_tensor(curr_choose_cls),
                 'support_names':support_names,
                 'query_name':curr_query_img_file_name.split('.')[0]
                 }

        # batch = {'query_img': torch.as_tensor(curr_query_img[:,:,0]),
        #          'query_mask': torch.as_tensor(curr_query_choose_mask[:,:,0]),

        #          'support_imgs': torch.as_tensor(support_imgs[:,:,:,0]),
        #          'support_masks': torch.as_tensor(support_masks[:,:,:,0]),

        #          'class_id': torch.tensor(curr_choose_support_cls)
        #          }

        return batch

if __name__ == "__main__":
    DatasetACDC()