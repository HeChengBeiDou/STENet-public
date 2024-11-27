import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# import augmentations
# from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import center_of_mass, rotate

# 图片形式加载数据

class BaseDataSets(Dataset): # base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([RandomGenerator(args.patch_size])
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform # 数据增强方式
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        
        ## 监督学习必须保证self.sample_list 只包含 img和mask都有的
        self.sample_list = self.file_exist(self.sample_list)

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def file_exist(self, sample_list):
        new_sample_list = []
        for case in sample_list:
            img_path = self._base_dir + "/data/imgs/{}.png".format(case)
            msk_path = self._base_dir + "/data/mask/{}.png".format(case)
            if os.path.exists(img_path) and os.path.exists(msk_path):
                new_sample_list.append(case)
        return new_sample_list

    def generate_distance_map(self, array, target_point, radius):
        """
        根据目标点生成距离映射数组，并保留大于0.5的值。

        参数：
        array (numpy.ndarray): 输入的全零二维数组。
        target_point (tuple): 目标点的坐标 (x, y)。
        radius (int): 影响范围的半径，决定有值的区域大小。

        返回：
        numpy.ndarray: 处理后的数组，只有值大于0.5的区域被保留。
        """
        # 获取数组的形状
        y_indices, x_indices = np.indices(array.shape)
        
        # 计算每个点到目标点的欧几里得距离
        distances = np.sqrt((x_indices - target_point[0]) ** 2 + (y_indices - target_point[1]) ** 2)
        
        # 使用半径进行归一化，并加快中心点附近的值变化
        normalized_distances = 1 - (distances / radius) ** 2
        
        # 保留距离在radius内的值，超出范围的值设为0
        result = np.where(distances <= radius, normalized_distances, 0)
        
        # 将目标点的值设为1
        result[target_point[1], target_point[0]] = 1
        
        # 将小于0.5的值设为0
        result = np.where(result > 0.5, result, 0)
        
        return result

    def compute_centroids(self, label_image, radius=10):
        classes = np.unique(label_image)
        centroids = {}
        centroids_2d = []
        for c in classes:
            if c == 0:  # 假设0是背景类，不计算质心
                mask = (label_image == c).astype(np.uint8)*0#这里必须是0 否则argmax求标签就全成0了
                curr_centroids = mask[0]
                centroids_2d.append(curr_centroids)
                continue
            mask = (label_image == c).astype(np.uint8)
            centroid = center_of_mass(mask)#(col,row)
            centroids[c] = centroid
            curr_centroids = np.zeros_like(label_image[0])
            # 使用 generate_distance_map 生成质心周围的渐变区域
            curr_centroids = self.generate_distance_map(curr_centroids, (int(centroid[2]), int(centroid[1])), radius)
            # curr_centroids[int(centroid[0]),int(centroid[1]), int(centroid[2])] = 1
            centroids_2d.append(curr_centroids)
        centroids_2d = np.array(centroids_2d)
        return centroids, centroids_2d

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        img_path = self._base_dir + "/data/imgs/{}.png".format(case)
        msk_path = self._base_dir + "/data/mask/{}.png".format(case)
        img = np.array(Image.open(img_path).convert('L'))/255 #RGB
        mask = np.array(Image.open(msk_path))
        # print(mask.max(), mask.min())
        # img = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE) / 255 # 以灰度图读入 并归一化除以255
        # mask = cv2.imread(self._base_dir + "/data/mask/{}.png".format(case), cv2.IMREAD_GRAYSCALE)  # 如果有标注 就返回只包含[0,1,2,3,4,5]的形状(H,W)的列表 如果没标注 就是全黑
    
        # h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        # image = h5f["image"][:]
        # label = h5f["label"][:]
          # 没有标签的label
        
        sample = img[:,:,None], mask[:,:,None]
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample) #############
        if self.split == "val" or self.split == "test":
            sample = self.transform(sample)
        new_img, new_mask = sample#先保存结果 如果没有问题 sample就不会变
        mask_np = new_mask.numpy()
        classes = np.unique(mask_np)#判断标签是否缺失
        
        if len(classes) == 6: #旋转后如果丢失标签 就不要旋转
            img, mask = sample
        else:
            from util.utils import myToTensor, myResize
            transformer = transforms.Compose([
                # myNormalize(datasets, train=False),
                myToTensor(),
                myResize(224, 224)
            ])
            sample = img[:,:,None], mask[:,:,None]#旋转后标签少了 就不要旋转
            sample = transformer(sample)
            img, mask = sample
            print(case)


        sample = {"image": img, "label": mask}#如果没有重新增强，就sample就没有变

        mask_np = mask.numpy()
        classes = np.unique(mask_np)
        # if len(classes) < 6:
        #     print(case, len(classes))
            
        #     return None
        centroids, centroids_2d = self.compute_centroids(mask_np, 4)#centroids_2d 是 (c,h,w) 只有类中心为1 坐标resize但没有归一化
        centroids_2d = torch.from_numpy(centroids_2d).to(dtype=torch.float32)
            
        sample["idx"] = idx
        sample["name"] = case
        sample["centroids_2d"] = centroids_2d
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None: ######## 
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label




def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


################# 数据增强
class RandomGenerator(object):
    def __init__(self, output_size): # args.patch_size:(256,256)
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label) #### label可以是None
        elif random.random() > 0.5:
            image, label = random_rotate(image, label) ###
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0) # zoom 用于缩放数组 后面是沿轴的缩放系数 order样条插值的顺序 0最邻近插值
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices # 0-139
        self.secondary_indices = secondary_indices # 140-1311
        self.secondary_batch_size = secondary_batch_size # 12
        self.primary_batch_size = batch_size - secondary_batch_size # 12

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices) # 0-139随机打乱
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable) # np.random.permutation 对数组进行随机排列


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
