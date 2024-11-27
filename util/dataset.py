r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data import DataLoader
from albumentations import RandomRotate90,Resize
from albumentations.core.composition import Compose
from util.camus import DatasetCAMUS
from util.acdc import DatasetACDC


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'camus':DatasetCAMUS,
            'acdc':DatasetACDC
        }

        # cls.img_mean = [0.485, 0.456, 0.406]
        # cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = Compose([Resize(img_size, img_size),
                                            # transforms.ToTensor(),
                                            # transforms.Normalize(cls.img_mean, cls.img_std)
                                            ]
                                            )

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
