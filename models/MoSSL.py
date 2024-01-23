import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import sys
import numpy as np
from models.layers import MDA, GSSL, MSSL
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################
## Spacial-Attention (SA) Layer
########################################
class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.channels = channels
        self.Wq = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.FC = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())

    def forward(self, rep):
        query = self.Wq(rep).permute(0,2,4,3,1)
        key = self.Wk(rep).permute(0,2,4,3,1)
        value = self.Wv(rep).permute(0,2,4,3,1)
        attention = torch.matmul(query, key.transpose(3, 4))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep.permute(0,4,1,3,2))
        del query, key, value, attention
        return rep
    
########################################
## Modality-Attention (MA) Layer
########################################
class MA(nn.Module):
    def __init__(self, channels):
        super(MA, self).__init__()
        self.channels = channels
        self.Wq = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())
        self.FC = nn.Sequential(
            nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1)),
            nn.ReLU())

    def forward(self, rep):
        query = self.Wq(rep).permute(0,3,4,2,1)
        key = self.Wk(rep).permute(0,3,4,2,1)
        value = self.Wv(rep).permute(0,3,4,2,1)
        attention = torch.matmul(query, key.transpose(3, 4))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep.permute(0,4,3,1,2))
        del query, key, value, attention
        return rep  
    
########################################
## Residual Block
########################################
class ResidualBlock(nn.Module):
    def __init__(self, num_modals, num_nodes, channels, dilation, kernel_size):
        super(ResidualBlock, self).__init__()
        self.num_modals = num_modals
        self.num_nodes = num_nodes
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num = 3        
        # Spatial-Attention Layer
        self.sa = SA(self.channels)
        # Modality-Attention Layer
        self.ma = MA(self.channels)
        # Temporal Convolution
        self.filter_convs = nn.Conv3d(in_channels = self.num * self.channels, 
                                      out_channels = self.num_modals * self.channels, 
                                      kernel_size = (self.num_modals, 1, self.kernel_size),
                                      dilation=(1,1,self.dilation))
        self.gate_convs = nn.Conv3d(in_channels = self.num * self.channels, 
                                    out_channels = self.num_modals * self.channels, 
                                    kernel_size = (self.num_modals, 1, self.kernel_size),
                                    dilation=(1,1,self.dilation))
        self.residual_convs = nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1))
        # Skip Connection
        self.skip_convs = nn.Conv3d(in_channels = self.channels, out_channels = self.channels, kernel_size = (1,1,1))

    def forward(self, rep):
        rep_list = []
        # Spatial-Attention Layer
        rep_spa = self.sa(rep)
        rep_list.append(rep_spa)
        # Modality-Attention Layer
        rep_sou = self.ma(rep)
        rep_list.append(rep_sou)
        rep_list.append(rep)
        rep = torch.cat(rep_list, dim=1) 
        # Temporal Convolution (TC)
        filter = self.filter_convs(rep)
        b, _, _, n, t = filter.shape
        filter = torch.tanh(filter).reshape(b, -1, self.num_modals, n, t)
        gate = self.gate_convs(rep)
        gate = torch.sigmoid(gate).reshape(b, -1, self.num_modals, n, t)
        rep = filter * gate
        # Parametrized skip connection
        save_rep = rep
        sk = rep
        sk = self.skip_convs(sk)
        rep = self.residual_convs(rep)
        return rep, sk, gate
    
########################################
## MoST Encoder
########################################    
class MoST_Encoder(nn.Module):
    def __init__(self, layers, num_modals, num_nodes, channels, kernel_size):
        super(MoST_Encoder, self).__init__()
        self.layers = layers
        # Residual Blocks
        self.residualblocks = nn.ModuleList()
        dilation = 1
        for i in range(self.layers):
            self.residualblocks.append(ResidualBlock(num_modals, num_nodes, channels, dilation, kernel_size))
            dilation *= 2
        
    def forward(self, rep):
        skip = 0        
        for i in range(self.layers):           
            residual = rep
            rep, sk, gate = self.residualblocks[i](rep)
            rep = rep + residual[:, :, :, :, -rep.size(4):]
            try:
                skip = sk + skip[:, :, :, :, -sk.size(4):]
            except:
                skip = sk
        return skip
    
########################################
## MoSSL Framework
######################################## 
class MoSSL(nn.Module):
    def __init__(self, device, num_comp, num_nodes, num_modals, n_his, n_pred, channels, layers, in_dim, kernel_size=2):
        super(MoSSL, self).__init__()
        # Linear Projection
        self.proj1 = nn.Sequential(
            nn.Conv3d(in_channels = in_dim, out_channels = channels, kernel_size = (1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels = channels, out_channels = 2*channels, kernel_size = (1,1,1)),
            nn.ReLU()
        )        
        # MoST Encoder
        self.most_encoder = MoST_Encoder(layers, num_modals, num_nodes, 2*channels, kernel_size)
        # Predictor
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels = 2*channels,out_channels = 2*channels,kernel_size = (1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels = 2*channels, out_channels = n_pred, kernel_size = (1,1,1))
        )
        # Multi-modality Data Augmentation
        self.mda = MDA(device, 2*channels, 2*channels,n_his, num_nodes, num_modals)
        # Global Self-Supervised Learning (GSSL)
        self.in_features = num_nodes*num_modals
        self.gssl = GSSL(self.in_features, 2*channels, num_comp)
        # Modality Self-Supervised Learning (MSSL)
        self.mssl = MSSL(2*channels, num_nodes, num_modals, device)
        
    def forward(self, input):
        input = input.permute(0, 4, 3, 2, 1)
        # Up-stream: the original view
        # Init representation
        x = self.proj1(input)
        # MoST Encoder
        rep = self.most_encoder(x)
        pred = self.predictor(rep)
        # Down-stream: the augmented view
        # Generate the Multi-modality Data Augmentation
        x_aug = self.mda(x, rep)
        # Shared MoST Encoder
        rep_aug = self.most_encoder(x_aug)
        # Global Self-Supervised Learning (GSSL)
        gssl_loss = self.gssl(rep, rep_aug)
        # Modality Self-Supervised Learning (MSSL)
        mssl_loss = self.mssl(rep, rep_aug)
        loss = gssl_loss + mssl_loss
        return pred.permute(0, 1, 3, 2, 4), loss
