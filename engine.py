import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from util.utils import save_batch_imgs,calculate_metric_percase
np.set_printoptions(suppress=True, precision=4)


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer,device):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, batch in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        # images, targets = data
        images = batch['image']#(n,c,h,w)
        targets = batch['label'].clone()#(n,1,h,w)
        targets_cetorid = batch['centroids_2d']
        query_name = batch['name']

        images, targets = images.to(device=device).float(), targets.to(device=device).float()
        targets_cetorid = targets_cetorid.to(device=device).float()

        logits3, logits, logits_centroid, cetrois_loss_label, ass_final_loss, dot_res_loss, centroid_class_loss, alpha_beta_loss = model(images,targets,targets_cetorid)

        mask_true_onehot = torch.nn.functional.one_hot(targets.squeeze(1).long(), num_classes=config.num_classes).permute(0, 3, 1, 2).float() #(n,c,h,w)
        mask_true_onehot = mask_true_onehot.to(device=device)
        loss1 = criterion(logits.contiguous(), mask_true_onehot)
        loss3 = criterion(logits3.contiguous(), mask_true_onehot)#ceDice不需要用softmax
        a = torch.nn.CrossEntropyLoss(ignore_index=0)

        loss2 = a(logits_centroid.contiguous(), targets_cetorid.argmax(1))*0.5
        if epoch < 50:
            loss = loss1 + loss3 + loss2*0 + cetrois_loss_label*0 + ass_final_loss*0 + dot_res_loss*0 + centroid_class_loss*0 + alpha_beta_loss*0
        else:
            loss = loss1 + loss3 + loss2 + cetrois_loss_label*1 + ass_final_loss + dot_res_loss + centroid_class_loss + alpha_beta_loss
        # + dot_res_loss
        #   + cetrois_loss
        print(
            f"{loss1.item():.4f}",
            f"{loss2.item():.4f}",
            f"{cetrois_loss_label.item():.4f}",
            f"{ass_final_loss.item():.4f}",
            f"{dot_res_loss.item():.4f}",
            f"{centroid_class_loss.item():.4f}",
            f"{alpha_beta_loss.item():.4f}"
        )

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)
        # break

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    writer,
                    device):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    dice_list = []
    hd95_list = []
    confusion = np.zeros((config.num_classes, config.num_classes))
    with torch.no_grad():
        for batch in tqdm(test_loader):
            img = batch['image']#(n,c,h,w)
            msk = batch['label'].clone()#(n,1,h,w)
            targets_cetorid = batch['centroids_2d']
            query_name = batch['name']

            img, msk = img.to(device=device).float(), msk.to(device=device).float()
            targets_cetorid = targets_cetorid.to(device=device).float()
            logits3, logits, logits_centroid, cetrois_loss_label, ass_final_loss, dot_res_loss, centroid_class_loss, alpha_beta_loss = model(img,msk,targets_cetorid)
            mask_true_onehot = torch.nn.functional.one_hot(msk.squeeze(1).long(), num_classes=config.num_classes).permute(0, 3, 1, 2).float() #(n,c,h,w)
            mask_true_onehot = mask_true_onehot.to(device=device)
            loss1 = criterion(logits.contiguous(), mask_true_onehot)#ceDice不需要用softmax
            a = torch.nn.CrossEntropyLoss(ignore_index=0)

            loss2 = a(logits_centroid.contiguous(), targets_cetorid.argmax(1))*0.5
            loss3 = criterion(logits3.contiguous(), mask_true_onehot)
            if epoch < 50:
                loss = loss1 + loss3 + loss2*0 + cetrois_loss_label*0 + ass_final_loss*0 + dot_res_loss*0 + centroid_class_loss*0 + alpha_beta_loss*0
            else:
                loss = loss1 + loss3 + loss2 + cetrois_loss_label + ass_final_loss + dot_res_loss + centroid_class_loss + alpha_beta_loss
            # + dot_res_loss

            loss_list.append(loss.item())
            msk_npy = msk.cpu().detach().numpy()
            # gts.append(msk_npy)
            # if type(out) is tuple:
            #     out = out[0]
            
            out_bin = logits.argmax(1,keepdim=True)
            out_bin_npy = out_bin.cpu().detach().numpy()
            # preds.append(out_bin_npy)
            curr_confusion = confusion_matrix(msk_npy.reshape(-1), out_bin_npy.reshape(-1))
            confusion += curr_confusion


            curr_dice_list = []
            curr_hd95_list = []
            for b in range(len(out_bin_npy)):#测试集每个图都算
                for i in range(0, config.num_classes):
                    ## (dice, hd95)
                    dice_score, hd95_distance = calculate_metric_percase(out_bin_npy[b] == i, msk_npy[b] == i)
                    curr_dice_list.append(dice_score)
                    curr_hd95_list.append(hd95_distance)

                dice_list.append(curr_dice_list)
                hd95_list.append(curr_hd95_list)

                curr_dice_list = []
                curr_hd95_list = []

            # break
    # preds = np.concatenate(preds, axis=0)
    # gts = np.concatenate(gts, axis=0)
    if epoch % config.val_interval == 0:
        # y_pre = np.array(preds).reshape(-1).astype(np.uint8)
        # y_true = np.array(gts).reshape(-1).astype(np.uint8)

        # y_pre = np.where(preds>=config.threshold, 1, 0)
        # y_true = np.where(gts>=0.5, 1, 0)

        # confusion = confusion_matrix(y_true, y_pre)
        
        # 计算总体精度
        accuracy = np.trace(confusion) / np.sum(confusion) if np.sum(confusion) != 0 else 0

        # 初始化变量
        num_classes = confusion.shape[0]
        sensitivity = np.zeros(num_classes)
        specificity = np.zeros(num_classes)
        f1_or_dsc = np.zeros(num_classes)
        miou = np.zeros(num_classes)

        # 逐类计算指标
        for i in range(num_classes):
            TP = confusion[i, i]
            FP = np.sum(confusion[:, i]) - TP
            FN = np.sum(confusion[i, :]) - TP
            TN = np.sum(confusion) - (TP + FP + FN)
            
            sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc[i] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou[i] = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        # 计算每个指标的平均值（宏平均）
        avg_sensitivity = np.mean(sensitivity[1:])
        avg_specificity = np.mean(specificity[1:])
        avg_f1_or_dsc = np.mean(f1_or_dsc[1:])
        avg_miou = np.mean(miou[1:])

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {avg_miou :.4f}, f1_or_dsc: {avg_f1_or_dsc :.4f}, accuracy: {accuracy :.4f}, \
                specificity: {avg_specificity :.4f}, sensitivity: {avg_sensitivity :.4f}, \n confusion_matrix: \n {confusion}'
        print(log_info)
        logger.info(log_info)

        mean_batch_dice_list = np.array(dice_list).mean(axis=0)
        mean_batch_hd95_list = np.array(hd95_list).mean(axis=0)

        log_info = f'dice_list: {np.mean(mean_batch_dice_list[1:]):.4f} \n {mean_batch_dice_list} \n mIoU: {np.mean(miou[1:]):.4f} \n {miou} \n  hd95_list: {np.mean(mean_batch_hd95_list[1:]):.4f} \n {mean_batch_hd95_list}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
        
    writer.add_scalar('val_loss', np.mean(loss_list), global_step=epoch)
    return np.mean(dice_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    device="cpu"):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    dice_list = []
    hd95_list = []
    confusion = np.zeros((config.num_classes, config.num_classes))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            img = batch['image']#(n,c,h,w)
            msk = batch['label'].clone()#(n,1,h,w)
            targets_cetorid = batch['centroids_2d']
            name = batch['name']

            img, msk = img.to(device=device).float(), msk.to(device=device).float()

            targets_cetorid = targets_cetorid.to(device=device).float()
            logits3, logits, logits_centroid, cetrois_loss_label, ass_final_loss, dot_res_loss, centroid_class_loss, alpha_beta_loss = model(img,msk,targets_cetorid)
            mask_true_onehot = torch.nn.functional.one_hot(msk.squeeze(1).long(), num_classes=config.num_classes).permute(0, 3, 1, 2).float() #(n,c,h,w)
            mask_true_onehot = mask_true_onehot.to(device=device)
            loss1 = criterion(logits.contiguous(), mask_true_onehot)#ceDice不需要用softmax
            a = torch.nn.CrossEntropyLoss(ignore_index=0)
            tmp = targets_cetorid.cpu().detach().numpy().astype(np.float16)
            other_tmp = logits_centroid.cpu().detach().numpy().astype(np.float16)

            tmp1 = tmp[0,1]
            tmp2 = tmp[0,2]
            tmp3 = tmp[0,3]
            tmp4 = tmp[0,4]
            tmp5 = tmp[0,5]
            loss2 = a(logits_centroid.contiguous(), targets_cetorid.argmax(1))*0.5
            loss3 = criterion(logits3.contiguous(), targets_cetorid.argmax(1))
            loss = loss1 + loss3 + loss2 + cetrois_loss_label + ass_final_loss + dot_res_loss + centroid_class_loss + alpha_beta_loss
            # + dot_res_loss

            loss_list.append(loss.item())

            msk_npy = msk.cpu().detach().numpy()
            # gts.append(msk_npy)
            # if type(out) is tuple:
            #     out = out[0]
            
            out_bin = logits.argmax(1,keepdim=True)
            out_bin_npy = out_bin.cpu().detach().numpy()
            # y_pre = np.array(preds).reshape(-1).astype(np.uint8)
            # y_true = np.array(gts).reshape(-1).astype(np.uint8)
            curr_confusion = confusion_matrix(msk_npy.reshape(-1), out_bin_npy.reshape(-1))
            confusion += curr_confusion
            # preds.append(out_bin_npy) 


            curr_dice_list = []
            curr_hd95_list = []
            for b in range(len(out_bin_npy)):#测试集每个图都算
                for i in range(0, config.num_classes):
                    ## (dice, hd95)
                    dice_score, hd95_distance = calculate_metric_percase(out_bin_npy[b] == i, msk_npy[b] == i)
                    curr_dice_list.append(dice_score)
                    curr_hd95_list.append(hd95_distance)

                dice_list.append(curr_dice_list)
                hd95_list.append(curr_hd95_list)

                curr_dice_list = []
                curr_hd95_list = []

            ## TODO 这里应该有每张图的精度 例如miou 输出列表即可

            ## save_imgs
            # save_path = config.work_dir + 'outputs/'
            # save_batch_imgs(imgs=img, msks=msk, msk_preds=out_bin, save_path=save_path, test_data_name=name)

            # break

        # preds = np.concatenate(preds, axis=0)
        # gts = np.concatenate(gts, axis=0)
        # y_pre = np.array(preds).reshape(-1).astype(np.uint8)
        # y_true = np.array(gts).reshape(-1).astype(np.uint8)

        # confusion = confusion_matrix(y_true, y_pre)
        
        # 计算总体精度
        accuracy = np.trace(confusion) / np.sum(confusion) if np.sum(confusion) != 0 else 0

        # 初始化变量
        num_classes = confusion.shape[0]
        sensitivity = np.zeros(num_classes)
        specificity = np.zeros(num_classes)
        f1_or_dsc = np.zeros(num_classes)
        miou = np.zeros(num_classes)

        # 逐类计算指标
        for i in range(num_classes):
            TP = confusion[i, i]
            FP = np.sum(confusion[:, i]) - TP
            FN = np.sum(confusion[i, :]) - TP
            TN = np.sum(confusion) - (TP + FP + FN)
            
            sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc[i] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou[i] = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        # 计算每个指标的平均值（宏平均）
        avg_sensitivity = np.mean(sensitivity[1:])
        avg_specificity = np.mean(specificity[1:])
        avg_f1_or_dsc = np.mean(f1_or_dsc[1:])
        avg_miou = np.mean(miou[1:])


        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {avg_miou :.4f}, f1_or_dsc: {avg_f1_or_dsc :.4f}, accuracy: {accuracy :.4f}, \
                specificity: {avg_specificity :.4f}, sensitivity: {avg_sensitivity :.4f}, \n confusion_matrix: \n {confusion}'
        print(log_info)
        logger.info(log_info)

        ## TODO  这里可以看每一个图的dice 如果结果不好可以看 平时没有必要
        mean_batch_dice_list = np.array(dice_list).mean(axis=0)
        mean_batch_hd95_list = np.array(hd95_list).mean(axis=0)

        log_info = f'dice_list: {np.mean(mean_batch_dice_list[1:]):.4f} \n {mean_batch_dice_list} \n mIoU: {np.mean(miou[1:]):.4f} \n {miou} \n  hd95_list: {np.mean(mean_batch_hd95_list[1:]):.4f} \n {mean_batch_hd95_list}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)