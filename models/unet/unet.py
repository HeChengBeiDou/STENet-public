""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.unet.unet_parts import *
except:
    from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        ## 质心预测图
        self.up1_centroid = Up(1024, 512 // factor, bilinear)
        self.up2_centroid = Up(512, 256 // factor, bilinear)
        self.up3_centroid = Up(256, 128 // factor, bilinear)
        self.up4_centroid = Up(128, 64, bilinear)
        self.outc_centroid = OutConv(64, n_classes)


        self.final_conv1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
        self.final_conv2 = nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1)
        self.final_conv3 = nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1)
        self.alpha_se = PoolConvFC(self.n_classes-1, self.n_classes-1, 224, 224)# 池化卷积FC得到坐标
        self.beta_se = PoolConvFC(self.n_classes-1, self.n_classes-1, 224, 224)

        self.final_guide_conv = nn.Conv2d(self.n_classes*2, self.n_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.CEloss = nn.CrossEntropyLoss()

    def find_center_coordinates(self, x, percentile=20):
        """
        通过差分计算梯度找到候选点，并从候选点中根据像素值挑选出中心点。
        这里使用每个通道的非零梯度的指定百分位数作为阈值筛选候选点。

        参数:
            x: torch.Tensor 形状为 (n, c, h, w)
            percentile: 用于筛选候选点的梯度百分位数，默认为 20%

        返回:
            center_coords: torch.Tensor 形状为 (n, c, 2)，其中最后一维是 (y, x) 坐标
        """
        n, c, h, w = x.shape
        
        # 计算差分梯度，沿y和x方向
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]  # 沿y方向的梯度
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]  # 沿x方向的梯度
        
        # 填充梯度以匹配原始张量的大小
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 1, 0))  # 在y方向顶部填充
        grad_x = torch.nn.functional.pad(grad_x, (1, 0, 0, 0))  # 在x方向左侧填充

        # 计算梯度的范数
        grad_magnitude = torch.sqrt(grad_y**2 + grad_x**2)
        
        # 初始化候选掩码
        candidate_mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
        
        for i in range(n):  # 对每个样本进行处理
            for j in range(c):  # 对每个通道进行处理
                grad_channel = grad_magnitude[i, j].view(-1)  # 展平成一维向量
                
                # 取出非零梯度值
                non_zero_grads = grad_channel[grad_channel > 0]
                
                if non_zero_grads.numel() > 0:  # 如果存在非零梯度
                    # 计算该通道的百分位数阈值
                    threshold = torch.quantile(non_zero_grads, percentile / 100.0)
                else:
                    threshold = torch.tensor(0.0).to(x.device)  # 若全为零梯度，设阈值为 0
                
                # 根据该通道的阈值生成候选点掩码
                candidate_mask[i, j] = grad_magnitude[i, j] < threshold
        
        # 使用候选掩码提取候选点的像素值
        candidate_values = torch.where(candidate_mask, x, torch.tensor(float('-inf')).to(x.device))
        
        # 从候选点中找到最大值的位置
        max_vals, max_indices = torch.max(candidate_values.view(n, c, -1), dim=-1)
        
        # 将一维索引转换为二维 (y, x) 坐标
        max_y_coords = max_indices // w
        max_x_coords = max_indices % w
        
        # 归一化坐标
        max_y_coords = max_y_coords / h
        max_x_coords = max_x_coords / w
        
        # 将 (y, x) 坐标组合起来
        center_coords = torch.stack((max_y_coords, max_x_coords), dim=-1).to(dtype=torch.float32)

        return center_coords

    def find_max_coordinates(self, x):
        """
        根据输入张量 (n, c, h, w) 找到每个样本每个通道中最大值的横纵坐标

        参数:
            x: torch.Tensor 形状为 (n, c, h, w)

        返回:
            max_coords: torch.Tensor 形状为 (n, c, 2)，其中最后一维是 (y, x) 坐标
        """
        n, c, h, w = x.shape
        
        # 找到每个通道的最大值
        max_vals, max_indices = torch.max(x.view(n, c, -1), dim=-1)
        
        # 将一维索引转换为二维 (y, x) 坐标
        max_y_coords = max_indices // w
        max_x_coords = max_indices % w
        
        max_y_coords = max_y_coords / h
        max_x_coords = max_x_coords / w

        # 将 (y, x) 坐标组合起来
        max_coords = torch.stack((max_y_coords, max_x_coords), dim=-1).to(dtype=torch.float32)
        
        return max_coords

    def final_out(self, Y, C):#这个必须改 能够用质心图指导分割结果变好
        # enc_Y = self.attention(Y, C)
        enc_cat_Y = torch.cat([Y, C], dim=1)
        out = self.final_guide_conv(enc_cat_Y) + Y #(n,c,h,w)
        out = self.final_conv1(self.pool(out))

        Y_down = self.pool(Y)
        out_cat = torch.cat([out, Y_down], dim=1)
        out = self.final_conv2(out_cat)
        out = self.up(out)
        out_cat = torch.cat([out, C], dim=1)
        out = self.final_conv3(out_cat)
        return out


    def forward(self, x, msk=None, centroid_2d=None):
        save_x = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        ## 质心预测分支 预测之后 用向量合成进行类间交互？ 求梯度 确保质心是局部最大 且向向量合成方向靠拢

        ## 初始化定义矢量规则集合
        

        x_centroid = self.up1_centroid(x5, x4)
        x_centroid = self.up2_centroid(x_centroid, x3)
        x_centroid = self.up3_centroid(x_centroid, x2)
        x_centroid = self.up4_centroid(x_centroid, x1)
        logits_centroid = self.outc_centroid(x_centroid)#直接和质心图算loss

        ## 根据质心图 计算每个中心点  最大值  通过组合规则计算loss  更新矢量规则R



        centroid_class = self.find_center_coordinates(logits_centroid)#(n,c,2) #这个通过centroid_class_loss更新
        LV = centroid_class[:,1]# 这个训练过程必须算logits_centroid的
        LA = centroid_class[:,2]
        RV = centroid_class[:,3]
        RA = centroid_class[:,4]
        DAO = centroid_class[:,5]

        if centroid_2d is not None:
            centroid_class_label = self.find_max_coordinates(centroid_2d)#(n,c,2)

            LV_label = centroid_class_label[:,1]# 这个训练过程必须算logits_centroid的
            LA_label = centroid_class_label[:,2]
            RV_label = centroid_class_label[:,3]
            RA_label = centroid_class_label[:,4]
            DAO_label = centroid_class_label[:,5]

            centroid_class_loss = F.mse_loss(centroid_class, centroid_class_label)
        else:
            centroid_class_loss = torch.tensor(0).to(device=logits.device)

        ##这里先用logits_centroid+fc 更新alpha beta 但是不能更新过大
        #TODO 这里也得算一个alpha_m_label
        alpha_m = self.alpha_se(logits_centroid[:,1:])
        beta_m = self.beta_se(logits_centroid[:,1:])

        if centroid_2d is not None:
            alpha_label = self.alpha_se(centroid_2d[:,1:])
            beta_label = self.beta_se(centroid_2d[:,1:])

            alpha_beta_loss = F.mse_loss(alpha_label, alpha_m) + F.mse_loss(beta_label, beta_m)
        else:
            alpha_beta_loss = torch.tensor(0).to(device=logits.device)
        
        alpha_m = torch.mean(alpha_m, dim=0)
        beta_m  = torch.mean(beta_m, dim=0)


        if centroid_2d is not None:
            alpha_label = torch.mean(alpha_label, dim=0)
            beta_label = torch.mean(beta_label, dim=0)
            new_LV_label = (alpha_label[0])*(LA_label - RA_label) + (beta_label[0])*(RV_label - RA_label)+RA_label
            new_LA_label = (alpha_label[1])*(LV_label - RV_label) + (beta_label[1])*(RA_label - RV_label)+RV_label#alpha2(LV - RV) + beta2(RA - RV)
            new_RV_label = (alpha_label[2])*(LV_label - LA_label) + (beta_label[2])*(RA_label - LA_label)+LA_label## RV = alpha3(LV - LA) + beta3(RA - LA)
            new_RA_label=(alpha_label[3])*(RV_label - LV_label) + (beta_label[3])*(LA_label - LV_label)+LV_label## RA = alpha4(RV - LV) + beta4(LA - LV)
            new_DAO_label = (alpha_label[4])*(LA_label - RA_label) + (beta_label[4])*(LA_label - LV_label)+LA_label# DAO = alpha5(LA - RA) + beta5(LA - LV)

            cetrois_loss_label = (F.mse_loss(new_RA_label, RA_label) + F.mse_loss(new_RV_label, RV_label) + F.mse_loss(new_LA_label, LA_label)+F.mse_loss(new_LV_label, LV_label)+F.mse_loss(new_DAO_label, DAO_label))/5/RA_label.numel() #这个得到正值
        else:
            cetrois_loss_label = torch.tensor(0).to(device=logits.device)

        #这个是消融实验 证明是不是得到了准确的相对位置关系 因为不准 所以训练过程不用
        new_LV = (alpha_m[0])*(LA - RA) + (beta_m[0])*(RV - RA)+RA
        new_LA = (alpha_m[1])*(LV - RV) + (beta_m[1])*(RA - RV)+RV#alpha2(LV - RV) + beta2(RA - RV)
        new_RV = (alpha_m[2])*(LV - LA) + (beta_m[2])*(RA - LA)+LA## RV = alpha3(LV - LA) + beta3(RA - LA)
        new_RA=(alpha_m[3])*(RV - LV) + (beta_m[3])*(LA - LV)+LV## RA = alpha4(RV - LV) + beta4(LA - LV)
        new_DAO = (alpha_m[4])*(LA - RA) + (beta_m[4])*(LA - LV)+LA# DAO = alpha5(LA - RA) + beta5(LA - LV)

        cetrois_loss = (F.mse_loss(new_RA, RA) + F.mse_loss(new_RV, RV) + F.mse_loss(new_LA, LA)+F.mse_loss(new_LV, LV)+F.mse_loss(new_DAO, DAO))/5/RA.numel() #这个不用          
        
        final_logits_pre = self.final_out(logits, logits_centroid)
        ## 局部注意力 
        if centroid_2d is not None:
            final_logits_label = self.final_out(logits, centroid_2d)
            
            ## TODO  这两个要相似
            ass_final_loss = F.mse_loss(final_logits_label, final_logits_pre)
        else:
            ass_final_loss = torch.tensor(0).to(device=logits.device)

        epsilon = 1e-8  # 防止分母为0
        dot_res_loss = 1 / (torch.mean(logits_centroid * logits) + epsilon)#质心图和分割图点乘 理想情况下sum达到最大 最差情况一点没对上 sum为0   这个得到负值

        return final_logits_pre, logits, logits_centroid, 1*cetrois_loss_label, 0*ass_final_loss, 0*dot_res_loss, 0*centroid_class_loss, 0*alpha_beta_loss





if __name__ == "__main__":
    n_channels=1
    n_classes=6
    net = UNet(n_channels=n_channels, n_classes=n_classes,bilinear=True)
    # net.apply(init)## apply方法是内置的 可以不重写
    data = torch.rand((2, 1, 224, 224))#(n,c,h,w)
    final_out, out, out_logits_centroid, cetrois_loss, ass_final_loss, dot_res_loss, centroid_class_loss, alpha_beta_loss = net(data)
    print(out.shape)
    print(out_logits_centroid.shape)
    

    # 计算网络参数
    # print('net total parameters:', sum(param.numel() for param in net.parameters()))
