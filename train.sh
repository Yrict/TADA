if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

dataset_path=./dataset/
patience=15
imu_len=200 # imu desired_length
au_len=60 #au
c_in=12 # 没用
c_out=14
d_model=256
n_heads=8
e_layers=6
d_layers=6
d_ff=512
conv_method=conv
tf_dropout=0.1
conv_dropout=0.5
user=xxx
num_worker=16
train_epochs=200
batch_size=32
learning_rate=0.0001
dataset_type=still
total_people=6
pred_len=45
user_mode=within
spilt_mode=cut
time_run=00001251
lambda1=1
lambda2=10
lstm_bidirection=True
seq_type=transformer
cuda="0"
devices="3,4"
pos_e=encoder
# is-generate
# for user in wzf yty yzh lls
# do
python -u exp_main.py \
    --dataset_path $dataset_path \
    --imu_len $imu_len\
    --au_len $au_len\
    --c_in $c_in\
    --c_out $c_out\
    --d_model $d_model\
    --n_heads $n_heads\
    --e_layers $e_layers\
    --d_layers $d_layers\
    --d_ff $d_ff\
    --conv_dropout $conv_dropout\
    --tf_dropout $tf_dropout\
    --train_epochs $train_epochs\
    --batch_size $batch_size\
    --patience $patience\
    --num_worker $num_worker\
    --learning_rate $learning_rate\
    --user_mode $user_mode\
    --spilt_mode $spilt_mode\
    --time_run $time_run\
    --total_people $total_people\
    --cuda $cuda\
    --lambda1 $lambda1\
    --lambda2 $lambda2\
    --conv_method $conv_method\
    --pos_e $pos_e\
    --user $user\
    --use_multi_gpu False\
    --lstm_bidirection $lstm_bidirection\
    --seq_type $seq_type\
    --pred_len $pred_len\
    --devices $devices\
    --dataset_type $dataset_type >logs/$time_run'_'$seq_type'_'$lstm_bidirection'_'$user'_'$conv_method'_pred'$pred_len'_cdo'$conv_dropout'_fdo'$tf_dropout'_'$lambda1'_'$lambda2'_im'$imu_len'_au'$au_len'_ci'$c_in'_co'$c_out'_lr'$learning_rate'_dm'$d_model'_nh'$n_heads'_el'$e_layers'_dl'$d_layers'_dff'$d_ff'_'$user_mode'_'$spilt_mode'_'$batch_size'_'$dataset_type.log 
# done
