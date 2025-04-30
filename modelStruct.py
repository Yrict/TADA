import torch
import torch.nn as nn
import torch.nn.functional as F
from AUPredictor import EnDecoder,Encoder,EmbedDecoder,ChannelAttention,ConvBlock2
from DACT import Seq_Transformer
class TADA(nn.Module):

    def __init__(self, conv_method:str='conv',conv_dropout:float=0., dec_dropout:float=0.,enc_dropout:float=0.,e_layers:int=1,
                   d_layers:int=1, pe:str='zeros', n_heads:int=16, learn_pe:bool=True,
                   d_ff:int=512,d_model:int=128, au_len:int=60,imu_len:int=200, c_in:int=12,
                   c_out:int=14,attn_dropout:float=0.,pos_e:str='encoder',
                   seq_d_model:int=256, seq_type:str='lstm',lstm_bidirection:bool=False):
        super(TADA, self).__init__()

        mid_channel =128
        self.lstm_bidirection = lstm_bidirection
        # conv_block
        self.conv_block = ConvBlock2(kernel_size=9,window=imu_len,dropout=conv_dropout,channel=mid_channel)
        self.fc2 = nn.Linear(mid_channel, c_out)  
        self.leaky_relu = nn.LeakyReLU()
        self.channel_attention = ChannelAttention(input_channels=imu_len, d_model=mid_channel, time_len=c_in)
        self.conv_method = conv_method
        self.seq_type = seq_type

        # decoder
        if pos_e == 'encoder':
            self.decoder = EnDecoder(in_channel=mid_channel,sq_len=au_len,d_model=d_model,d_layers=d_layers,dropout=dec_dropout,n_heads=n_heads,d_ff=d_ff)
        elif pos_e == 'embed':
            self.decoder = EmbedDecoder(c_out=c_out,d_model=d_model,d_layers=d_layers,dropout=dec_dropout,n_heads=n_heads,d_ff=d_ff)
        self.projection = nn.Linear(d_model, c_out, bias=True)

        # encoder

        self.encoder = Encoder(in_channel=c_out, dropout=enc_dropout, sq_len=au_len, n_layers=e_layers,d_model=d_model,n_heads=n_heads,
                        d_ff=d_ff)
        self.seq_transformer = Seq_Transformer(patch_size=mid_channel, dim=seq_d_model, depth=4, heads=4, mlp_dim=64)
        if self.lstm_bidirection:
            self.lstm = nn.LSTM(input_size=mid_channel, hidden_size=128,  bidirectional = True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=mid_channel, hidden_size=256, batch_first=True)
        
    def featureExtractor(self, x):
        features = []
        #  x:[batch, au_len(e.g.60), imu_len(e.g.200), channels]
        x = x.permute(0,1,3,2)
        if self.conv_method == 'attention':
            for i in range(x.shape[1]):
                features.append(self.channel_attention(x[:,i,:,:])) # [batch, imu_len, channels]
        else:
            for i in range(x.shape[1]):
                features.append(self.conv_block(x[:,i,:,:])) # [batch, imu_len, channels]
        features_stacked = torch.stack(features, dim=0) #  [au_len, batch, channels]
        output = features_stacked.permute(1,0,2) #  [batch, au_len, channels]
        return output
    
    def transformer(self, dec_in, enc_in):
        enc_outputs, enc_self_attns = self.encoder(enc_in)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_in, enc_outputs)

        output = self.projection(dec_outputs) # x: [bs  x seq_len x nvars]
        return output
    
    def forward(self, still_x, walk_x, still_enc_inp, walk_enc_inp):

        # still_x
        still_feature = self.featureExtractor(still_x)
        walk_feature = self.featureExtractor(walk_x)
        # normalize projection feature vectors
        still_feature = F.normalize(still_feature, dim=1)
        walk_feature = F.normalize(walk_feature, dim=1)

        # branch one
        if self.seq_type == 'lstm':
            still_output, (still_h_n, c_n) = self.lstm(still_feature)
            walk_output, (walk_h_n, c_n)  = self.lstm(walk_feature)
            if self.lstm_bidirection:
                h_forward = still_h_n[-2, :, :]  # the last layer's forward hidden state
                h_backward = still_h_n[-1, :, :] # the last layer's backward hidden state
                still_cls = torch.cat((h_forward, h_backward), dim=1)
                h_forward = walk_h_n[-2, :, :]  
                h_backward = walk_h_n[-1, :, :]  
                walk_cls = torch.cat((h_forward, h_backward), dim=1)
            else:
                still_cls = still_h_n[-1,:,:] 
                walk_cls = walk_h_n[-1,:,:]

        elif self.seq_type == 'transformer':
            still_cls = self.seq_transformer(still_feature) 
            walk_cls = self.seq_transformer(walk_feature)
        # input: (batch_size, au_len, channels)
        # cls : (batch_size, channels)


        # branch two
        still_output = self.transformer(still_feature, still_enc_inp)
        walk_output = self.transformer(walk_feature, walk_enc_inp)
        # x_dec -> [ batch, au_len,  channels]
        # dec_in -> [ batch, au_len,  au_channels]

        return still_output, walk_output, still_cls, walk_cls