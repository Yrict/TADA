import os
import torch.nn as nn
import time
from torch.optim import lr_scheduler
from utils import *
import torch
from torch.utils.tensorboard import SummaryWriter
from dataLoader import My_Dataset
def vali(args, model, still_loader, walk_loader, criterion, device, label, test_path = None):
    still_loss_list = []
    walk_loss_list = []
    total_loss = []
    content_loss_list = []
    walk_outputs_result = []
    still_outputs_result = []
    walk_labels = []
    still_labels = []
    model.eval()
    with torch.no_grad():
        for i, ((still_x, still_y), (walk_x, walk_y)) in enumerate(zip(still_loader, walk_loader)): # [batch, au_len,labels/imu_len,channel]
            still_x = still_x.float().to(device)
            still_y = still_y.float().to(device)

            walk_x = walk_x.float().to(device)
            walk_y = walk_y.float().to(device)

            walk_enc_inp = torch.zeros_like(walk_y[:, -args.pred_len:, :]).float()
            walk_enc_inp = torch.cat([walk_y[:, :(args.au_len - args.pred_len), :], walk_enc_inp], dim=1).float().to(device)

            still_enc_inp = torch.zeros_like(still_y[:, -args.pred_len:, :]).float()
            still_enc_inp = torch.cat([still_y[:, :(args.au_len - args.pred_len), :], still_enc_inp], dim=1).float().to(device)

            # encoder - decoder

            still_output, walk_output, still_cls, walk_cls = model(still_x, walk_x, still_enc_inp, walk_enc_inp)
                
            still_output = still_output.detach().cpu()
            walk_output = walk_output.detach().cpu()
            still_cls = still_cls.detach().cpu()
            walk_cls = walk_cls.detach().cpu()
            walk_y = walk_y.detach().cpu()
            still_y = still_y.detach().cpu()
            
            still_pred = still_output[:, -args.pred_len:, :]
            still_true = still_y[:, -args.pred_len:, :]

            walk_pred = walk_output[:, -args.pred_len:, :]
            walk_true = walk_y[:, -args.pred_len:, :]

            still_loss = criterion(still_pred, still_true)
            walk_loss = criterion(walk_pred, walk_true)

            content_loss = contentLoss(still_cls, walk_cls)

            loss = args.lambda1 * (still_loss + walk_loss) + args.lambda2 * content_loss

            still_loss_list.append(still_loss.item())
            walk_loss_list.append(walk_loss.item())
            total_loss.append(loss.item())
            content_loss_list.append(content_loss.item())
            walk_outputs_result.append(walk_output)
            still_outputs_result.append(still_output)
            walk_labels.append(walk_y)
            still_labels.append(still_y)
            if label == "test":
                if i % 50 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    gt = walk_y[3, :, :]
                    pd = walk_output[3, :,:]
                    visual_pred_lens(args.au_len, args.c_out, true=gt, preds=pd, name=os.path.join(test_path, 'walk_'+ str(i) + '.pdf'))
                    gt = still_y[3, :, :]
                    pd = still_output[3, :,:]
                    visual_pred_lens(args.au_len, args.c_out, true=gt, preds=pd, name=os.path.join(test_path, 'still_'+ str(i) + '.pdf'))

    total_loss = np.average(total_loss)
    walk_loss_list = np.average(walk_loss_list)
    still_loss_list = np.average(still_loss_list)
    content_loss_list = np.average(content_loss_list)
    
    walk_outputs_result = torch.stack(walk_outputs_result, dim=0).numpy()
    walk_labels = torch.stack(walk_labels, dim=0).numpy()

    walk_outputs_result = walk_outputs_result.reshape(-1, walk_outputs_result.shape[-2], walk_outputs_result.shape[-1])
    walk_labels = walk_labels.reshape(-1, walk_labels.shape[-2], walk_labels.shape[-1])

    walk_mae, walk_mse, rmse, walk_rse, corr = metric(walk_outputs_result, walk_labels)
    walk_channel_maes, walk_channel_mses = channel_mae(walk_outputs_result, walk_labels)
    # --------
    still_outputs_result = torch.stack(still_outputs_result, dim=0).numpy()
    still_labels = torch.stack(still_labels, dim=0).numpy()

    still_outputs_result = still_outputs_result.reshape(-1, still_outputs_result.shape[-2], still_outputs_result.shape[-1])
    still_labels = still_labels.reshape(-1, still_labels.shape[-2], still_labels.shape[-1])
    
    still_mae, still_mse, rmse, still_rse, corr = metric(still_outputs_result, still_labels)
    still_channel_maes, still_channel_mses = channel_mae(still_outputs_result, still_labels)

    if label == "test":
        print('Test: total loss {}, walk loss {}, still loss {}, nceloss {}'.format(total_loss, walk_loss_list, still_loss_list, content_loss_list))
        print('Test: --walk-- result mae:{}, mse:{}; --still-- result mae:{}, mse:{}'.format(walk_mae, walk_mse, still_mae, still_mse))
        print('Test --walk-- channels mae:{}, mse:{}; --still-- channels mae:{}, mse:{}'.format(walk_channel_maes, walk_channel_mses, still_channel_maes, still_channel_mses))
    elif label == "vali":
        print('Vali: total loss {}, walk loss {}, still loss {}, nceloss {}'.format(total_loss, walk_loss_list, still_loss_list, content_loss_list))
        print('Vali: --walk-- result mae:{}, mse:{}; --still-- result mae:{}, mse:{}'.format(walk_mae, walk_mse, still_mae, still_mse))
        print('Vali --walk-- channels mae:{}, mse:{}; --still-- channels mae:{}, mse:{}'.format(walk_channel_maes, walk_channel_mses, still_channel_maes, still_channel_mses))
    model.train()
    return total_loss, still_loss_list, walk_loss_list

