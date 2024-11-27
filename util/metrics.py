import torch
import numpy as np
from torch import Tensor
from typing import Tuple


class Metrics:
    def __init__(self, n_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.n_classes = n_classes
        self.hist = torch.zeros(n_classes, n_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.n_classes + pred[keep], minlength=self.n_classes**2).view(self.n_classes, self.n_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

## 来自SGCN
import torch
import numpy as np

## 下面包含两个同名函数 参数不同 调用的时候会自动选择 不会报错

def eval_metrics(output, target, n_classes,conf_matrix):
    '''
    这个会直接把结果累加到conf_matrix并且返回(因为conf_matrix是矩阵,所以返回不返回conf_matrix都已经被累加),也就是说会直接改变形参
    output (h, w) tensor
    target (h, w) tensor
    '''
    # if n_classes > 1:
    #     _, predict = output.max(1)#变成标签值
    # else:
    #     output
    predict = output
    c_predict=predict.cpu().numpy().flatten()#(n,)
    c_target=target.cpu().numpy().flatten()#(n,)
    for i in range(len(c_predict)):
        conf_matrix[c_predict[i],c_target[i]] += 1

    predict = predict.long() + 1
    target = target.long() + 1
    # predict = predict.long()
    # target = target.long()

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), n_classes, 1, n_classes)
    area_pred = torch.histc(predict.float(), n_classes, 1, n_classes)
    area_lab = torch.histc(target.float(), n_classes, 1, n_classes)
    area_union = area_pred + area_lab - area_inter


    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    #pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    #mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()
    return correct, labeld, inter, union,conf_matrix


def eval_metrics(output, target, n_classes):
    '''
    这个每一次就重新生成一个混淆矩阵 累加工作放到外面去
    output (h, w) tensor
    target (h, w) tensor
    '''
    conf_matrix = np.zeros((np.max((2,n_classes)),np.max((2,n_classes))))
    predict = output
    c_predict=predict.cpu().numpy().flatten()#(n,)
    c_target=target.cpu().numpy().flatten()#(n,)
    for i in range(len(c_predict)):
        conf_matrix[c_predict[i],c_target[i]] += 1

    predict = predict.long() + 1
    target = target.long() + 1
    # predict = predict.long()
    # target = target.long()

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), n_classes, 1, n_classes)
    area_pred = torch.histc(predict.float(), n_classes, 1, n_classes)
    area_lab = torch.histc(target.float(), n_classes, 1, n_classes)
    area_union = area_pred + area_lab - area_inter


    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    return correct, labeld, inter, union,conf_matrix

def eval_metrics_multiclass(output, target, n_classes,conf_matrix,contain_background=True):
    '''
    专用于多分类
    output (h, w) tensor
    target (h, w) tensor
    '''
    predict = output
    c_predict=predict.cpu().numpy().flatten()#(n,)
    c_target=target.cpu().numpy().flatten()#(n,)
    for i in range(len(c_predict)):
        conf_matrix[c_predict[i],c_target[i]] += 1

    ## 这里其实用+1来把背景类包含在计算过程中
    ## 如果去掉 +1 那就是不包含背景类 因为是多分类，所以去掉背景也有超过1类，所以仍旧可以计算
    if contain_background:
        ## 包含背景类
        predict = predict.long() + 1#[1,2]
        target = target.long() + 1#[1,2]
    else:
        # 去掉背景类
        predict = predict.long()#[1,2]
        target = target.long()#[1,2]

    pixel_labeled = (target > 0).sum()#总pixel数
    pixel_correct = ((predict == target)*(target > 0)).sum()#正确分类pixel数

    predict = predict * (target > 0).long()#这里target>0都成立 predict不变 [1,2]
    intersection = predict * (predict == target).long()#二维矩阵 包括0 1 2 代表交集
    
    # 如果包含背景类 那之前已经+1 所以背景类标签为1
    # 如果不包含背景类 那背景类标签仍为0
    # 所以那种情况下面的1和n_classes都不需要改变
    area_inter = torch.histc(intersection.float(), n_classes, 1, n_classes)
    area_pred = torch.histc(predict.float(), n_classes, 1, n_classes)
    area_lab = torch.histc(target.float(), n_classes, 1, n_classes)
    area_union = area_pred + area_lab - area_inter


    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    #pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    #mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()
    return correct, labeld, inter, union,conf_matrix

def eval_metrics_multiclass(output, target, n_classes,contain_background=True):
    '''
    专用于多分类
    output (h, w) tensor
    target (h, w) tensor
    '''
    conf_matrix = np.zeros((np.max((2,n_classes)),np.max((2,n_classes))))
    predict = output
    c_predict=predict.cpu().numpy().flatten()#(n,)
    c_target=target.cpu().numpy().flatten()#(n,)
    for i in range(len(c_predict)):
        conf_matrix[c_predict[i],c_target[i]] += 1

    ## 这里其实用+1来把背景类包含在计算过程中
    ## 如果去掉 +1 那就是不包含背景类 因为是多分类，所以去掉背景也有超过1类，所以仍旧可以计算
    if contain_background:
        ## 包含背景类
        predict = predict.long() + 1#[1,2]
        target = target.long() + 1#[1,2]
    else:
        # 去掉背景类
        predict = predict.long()#[1,2]
        target = target.long()#[1,2]

    pixel_labeled = (target > 0).sum()#总pixel数
    pixel_correct = ((predict == target)*(target > 0)).sum()#正确分类pixel数

    predict = predict * (target > 0).long()#这里target>0都成立 predict不变 [1,2]
    intersection = predict * (predict == target).long()#二维矩阵 包括0 1 2 代表交集
    
    # 如果包含背景类 那之前已经+1 所以背景类标签为1
    # 如果不包含背景类 那背景类标签仍为0
    # 所以那种情况下面的1和n_classes都不需要改变
    area_inter = torch.histc(intersection.float(), n_classes, 1, n_classes)
    area_pred = torch.histc(predict.float(), n_classes, 1, n_classes)
    area_lab = torch.histc(target.float(), n_classes, 1, n_classes)
    area_union = area_pred + area_lab - area_inter


    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    #pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    #mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()
    return correct, labeld, inter, union,conf_matrix