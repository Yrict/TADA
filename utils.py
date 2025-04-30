import numpy as np
import torch
import math
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))



def find_index(target,imu_time):
    index = 0
    while index < len(imu_time):
        if imu_time[index] >= target:
            break
        index += 1
    if index == len(imu_time) and imu_time[index-1] < target:
        return None
    return index


def plot_matrix(y_true, y_pred, labels_name, save_path,title=None, thresh=0.7, axis_labels=None):

    cm = confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)
    plt.yticks(num_local, axis_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
#             if int(cm[i][j] * 100 - 1) > 0:
            if i==j:
                print("waht",cm[i][j])
                plt.text(j, i, format(cm[i][j] * 100, '.0f') + '%',
                        ha="center", va="center",
#                         color="white")  
                        color="white" if cm[i][j] > thresh else "black")  
    fig=plt.gcf()
    plt.savefig(save_path,dpi=500, bbox_inches='tight')
    plt.show()

def fIAV(data):
    # - MAV - Mean Absolute Value
    # windows * sensor
    feature = []
    for i in range(data.shape[1]):
        feature.append(np.sum([abs(j-525) for j in data[:,i]])/data.shape[0])
    return feature

def fMAX(data):
    feature = []
    for i in range(data.shape[1]):
        feature.append(max(abs(data[:,i])))
    return feature

def fRMS(data):
    # - RMS - Root Mean Square 
    feature = []
    for i in range(data.shape[1]):
        square = [num*num for num in data[:,i]]
        feature.append(math.sqrt(np.mean(square)))
    return feature

def contentLoss(cls1, cls2):
    mse_loss_mean  = nn.MSELoss(reduction='mean')
    loss_mean = mse_loss_mean(cls1, cls2)
    return loss_mean


def nceLoss(cls1, cls2):
    # [batch, channels]
    # for i in np.arange(0, self.timestep):
    total = torch.mm(cls1, cls2.T)
    
    # Apply log softmax to each row
    log_softmax = nn.LogSoftmax(dim=1)
    
    # Extract the diagonal elements
    diagonal_elements = torch.diag(log_softmax(total))
    
    # Compute the loss
    nce = torch.sum(diagonal_elements)
    nce /= -1.0 * cls1.shape[0]
    return nce

def visual_pred_lens(pred_lens, output_channels, true=None, preds=None,  name='./pic/test.pdf'):
    """
    Results visualization
    """

    rows = 4 
    cols = output_channels // rows + (output_channels % rows > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 10)) 
    axs = axs.flatten()  
    x = np.arange(0, pred_lens)
    for i in range(output_channels):
        axs[i].plot(x, preds[:, i], label='Prediction', linewidth=2) 
        axs[i].plot(x, true[:, i], label='GroundTruth', linewidth=2) 
        axs[i].set_title(f'Channel {i+1}') 
        axs[i].legend()
        axs[i].set_ylim(0, 5) 

    # close axs that are not used
    for i in range(output_channels, rows*cols):
        fig.delaxes(axs[i])
    plt.legend()
    plt.tight_layout()  
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def channel_mae(y_pred, y_true):
    # calculate the absolute errors
    absolute_errors = np.abs(y_true - y_pred)
    # calculate the mean of the absolute errors
    channel_maes = np.mean(absolute_errors, axis=(0, 1))
    mse_errors = (y_pred - y_true) ** 2
    channel_mses = np.mean(mse_errors, axis=(0, 1))
    return channel_maes, channel_mses

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, rse, corr
