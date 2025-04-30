import torch
import os
import random
import argparse
import numpy as np
from modelStruct import TADA
from train import train


def main(opt):
    print('Args in experiment:')
    print(opt)

    # devices
    print("gpu_count",torch.cuda.device_count())
    opt.use_cuda = True if torch.cuda.is_available() and opt.use_cuda else False
    
    if opt.use_cuda and opt.use_multi_gpu:
        opt.devices = opt.devices.replace(' ', '')
        device_ids = opt.devices.split(',')
        print(device_ids)
        device_ids = [int(id_) for id_ in device_ids]
        opt.cuda = device_ids[0] # 主GPU
    if opt.use_cuda:
        device = torch.device('cuda:{}'.format(opt.cuda))
        print('device: ',device)
    else:
        device = torch.device('cpu')
        print('Use CPU')

    
    nw = min([os.cpu_count(), opt.batch_size, opt.num_worker if opt.batch_size > 1 else 0])
    print('Using {} dataloader workers every process'.format(nw))

    model = TADA(conv_dropout=args.conv_dropout,
                        conv_method=args.conv_method,
                        dec_dropout=args.tf_dropout,
                        enc_dropout=args.tf_dropout,
                        n_heads=args.n_heads,
                        e_layers=args.e_layers,
                        d_layers=args.d_layers,
                        d_ff=args.d_ff,
                        d_model=args.d_model,
                        au_len=args.au_len,
                        imu_len=args.imu_len,
                        c_in=args.c_in,
                        c_out=args.c_out,
                        pos_e=args.pos_e,
                        seq_type=args.seq_type,
                        lstm_bidirection=args.lstm_bidirection)
    dataset_start_name = '{}_P{}_{}_{}_w{}'.format(
        opt.dataset_type,
        opt.total_people,
        opt.user_mode,
        opt.spilt_mode,
        opt.au_len
    )
    for file_name in os.listdir(opt.dataset_path):
        if file_name.startswith(dataset_start_name) and file_name.endswith(str(opt.imu_len)):
            if opt.user in file_name:
                file_path = os.path.join(opt.dataset_path, file_name)
                walk_file_name = file_name.replace(opt.dataset_type, 'walk')
                walk_file_path = os.path.join(opt.dataset_path, walk_file_name)
                train(opt, file_path, walk_file_path, model, nw, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal-Aware Domain Adaptation for Noise Reduction in IMU-based Facial Perception')
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='root path of the data file')

    parser.add_argument('--imu_len', type=int, default=800, help='encoder input length') 
    parser.add_argument('--au_len', type=int, default=60, help='decoder input length')
    parser.add_argument('--c_in', type=int, default=12, help='input channels')
    parser.add_argument('--c_out', type=int, default=17, help='output channels')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--tf_dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--conv_dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--pos_e',type=str,default='encoder')
    parser.add_argument('--lambda1', type=float, default=0.5, help='lambda1 loss parameter')
    parser.add_argument('--lambda2', type=float, default=0.5, help='lambda2 loss parameter')
    parser.add_argument('--pred_len', type=int, default=1, help='pred au len of decoder')
    parser.add_argument('--conv_method',type=str,default='attention')
    parser.add_argument('--seq_type',type=str,default='transformer', help='transformer or lstm')
    parser.add_argument('--lstm_bidirection', type=lambda x: (str(x).lower() == 'true'), default=False)

    # optimization
    parser.add_argument('--num_worker', type=int, default=16, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--dataset_type',type=str,default='walk(moving) or still(static)')
    parser.add_argument('--total_people', type=int, default=8, help='total people of dataset')
    parser.add_argument('--user_mode', type=str, default='within', help='within or cross')
    parser.add_argument('--spilt_mode',type=str,default='cut', help='cut or random of dataset generation')
    parser.add_argument('--time_run',type=str,default='03071719', help='experiment label')
    parser.add_argument('--optimizer', type=str, default='adamw')  # sgd,adam,adamw

    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start') # 学习率上升部分所占比例
    parser.add_argument('--user',type=str,default='xxx')
    # GPU
    parser.add_argument('--use_cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--cuda',type=str,default='0')
    parser.add_argument('--use_multi_gpu',  type=lambda x: (str(x).lower() == 'true'), help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='4,6', help='device ids of multile gpus')
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    main(opt=args)