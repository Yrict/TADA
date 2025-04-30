import os
import torch.nn as nn
import time
from torch.optim import lr_scheduler
import random
from utils import *
import torch
from torch.utils.tensorboard import SummaryWriter
from dataLoader import My_Dataset
from vali import vali
from torch.utils.data import Subset


def train(args, file_still_path, file_walk_path, model, nw, device):
    path_label = '{}_{}_{}_lbi{}_pred{}_{}_{}_tdo{}_cdo{}_im{}_au{}_ci{}_co{}_lr{}_s{}_re{}_dm{}_nh{}_el{}_dl{}_dff{}_{}_{}_{}_{}.pkl'.format(
        args.time_run,
        args.conv_method,
        args.seq_type,
        args.lstm_bidirection,
        args.pred_len,
        args.lambda1,
        args.lambda2,
        args.tf_dropout,
        args.conv_dropout,
        args.imu_len,
        args.au_len,
        args.c_in,
        args.c_out,
        args.learning_rate,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.user_mode,
        args.spilt_mode,
        args.batch_size,
        args.dataset_type
    )
    log_path = os.path.join('./log/',args.time_run, args.user, path_label)
    # 创建日志文件
    tb_writer = SummaryWriter(log_dir=log_path)

    
    if args.use_multi_gpu and args.use_cuda:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)

    train_still_dataset = My_Dataset(path=file_still_path, label='train')
    val_still_dataset = My_Dataset(path=file_still_path, label='val')
    test_still_dataset = My_Dataset(path=file_still_path, label='test')


    train_walk_dataset = My_Dataset(path=file_walk_path, label='train')
    val_walk_dataset = My_Dataset(path=file_walk_path, label='val')
    test_walk_dataset = My_Dataset(path=file_walk_path, label='test')

    min_train = min(len(train_still_dataset), len(train_walk_dataset))
    min_val = min(len(val_still_dataset), len(val_walk_dataset))
    min_test = min(len(test_still_dataset), len(test_walk_dataset))

    # Randomly select min(nums of datasets) indices from the full dataset
    train_still_dataset_indices = random.sample(range(len(train_still_dataset)), min_train)
    train_still_subset_dataset = Subset(train_still_dataset, train_still_dataset_indices)
    train_walk_dataset_indices = random.sample(range(len(train_walk_dataset)), min_train)
    train_walk_subset_dataset = Subset(train_walk_dataset, train_walk_dataset_indices)

    val_still_dataset_indices = random.sample(range(len(val_still_dataset)), min_val)
    val_still_subset_dataset = Subset(val_still_dataset, val_still_dataset_indices)
    val_walk_dataset_indices = random.sample(range(len(val_walk_dataset)), min_val)
    val_walk_subset_dataset = Subset(val_walk_dataset, val_walk_dataset_indices)

    test_still_dataset_indices = random.sample(range(len(test_still_dataset)), min_test)
    test_still_subset_dataset = Subset(test_still_dataset, test_still_dataset_indices)
    test_walk_dataset_indices = random.sample(range(len(test_walk_dataset)), min_test)
    test_walk_subset_dataset = Subset(test_walk_dataset, test_walk_dataset_indices)


    train_still_loader = torch.utils.data.DataLoader(dataset=train_still_subset_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=nw, drop_last=True)
    vali_still_loader = torch.utils.data.DataLoader(dataset=val_still_subset_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=nw, drop_last=True)
    test_still_loader = torch.utils.data.DataLoader(dataset=test_still_subset_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=nw, drop_last=True)


    train_walk_loader = torch.utils.data.DataLoader(dataset=train_walk_subset_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=nw, drop_last=True)
    vali_walk_loader = torch.utils.data.DataLoader(dataset=val_walk_subset_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=nw, drop_last=True)
    test_walk_loader = torch.utils.data.DataLoader(dataset=test_walk_subset_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=nw, drop_last=True)
    
    assert len(train_walk_loader) == len(train_still_loader)
    assert len(vali_walk_loader) == len(vali_still_loader)
    assert len(test_walk_loader) == len(test_still_loader)
    print(f"len of train loader: {len(train_walk_loader)}, test loader{len(test_walk_loader)}, val loader{len(vali_walk_loader)}")
    
    time_now = time.time()
    train_steps = len(train_walk_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                        steps_per_epoch = train_steps,
                                        pct_start = args.pct_start,
                                        epochs = args.train_epochs,
                                        max_lr = args.learning_rate)
    # dir 
    weights_dir = os.path.join('./log/', args.time_run,args.user, path_label,"weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    test_path = os.path.join('./test_results/', args.time_run, args.user, path_label)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, ((still_x, still_y), (walk_x, walk_y)) in enumerate(zip(train_still_loader, train_walk_loader)): # [batch, au_len,labels/imu_len,channel]
            iter_count += 1
            model_optim.zero_grad()
            still_x = still_x.float().to(device)
            still_y = still_y.float().to(device)

            walk_x = walk_x.float().to(device)
            walk_y = walk_y.float().to(device)
            
            walk_enc_inp = torch.zeros_like(walk_y[:, -args.pred_len:, :]).float()
            walk_enc_inp = torch.cat([walk_y[:, :(args.au_len - args.pred_len), :], walk_enc_inp], dim=1).float().to(device)

            still_enc_inp = torch.zeros_like(still_y[:, -args.pred_len:, :]).float()
            still_enc_inp = torch.cat([still_y[:, :(args.au_len - args.pred_len), :], still_enc_inp], dim=1).float().to(device)


            still_output, walk_output, still_cls, walk_cls = model(still_x, walk_x, still_enc_inp, walk_enc_inp)

            still_pred = still_output[:, -args.pred_len:, :]
            still_true = still_y[:, -args.pred_len:, :]

            walk_pred = walk_output[:, -args.pred_len:, :]
            walk_true = walk_y[:, -args.pred_len:, :]

            still_loss = criterion(still_pred, still_true)
            walk_loss = criterion(walk_pred, walk_true)

            content_loss = contentLoss(still_cls, walk_cls)

            loss = args.lambda1 * (still_loss + walk_loss) + args.lambda2 * content_loss
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()
            scheduler.step()
                

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        valid_total, still_vali_loss, walk_vali_loss = vali(args, model, vali_still_loader, vali_walk_loader, criterion,device, "vali")
        test_total, still_test_loss, walk_test_loss = vali(args, model, test_still_loader, test_walk_loader, criterion, device, "test",test_path=test_path)# test visual
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali total Loss: {3:.7f} Still Vali Loss: {3:.7f} Walk Vali Loss: {3:.7f} | Test Total Loss: {3:.7f} Still Test Loss: {3:.7f} Walk Test Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, valid_total, still_vali_loss, walk_vali_loss, test_total, still_test_loss, walk_test_loss))
        
        early_stopping(valid_total, model, weights_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(model_optim, scheduler, epoch + 1, args)

    best_model_path = weights_dir + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model
