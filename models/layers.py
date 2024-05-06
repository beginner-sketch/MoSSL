import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import copy
import math
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
########################################
## Multi-modality Data Augmentation (MDA)
########################################    
class MDA(nn.Module):
    def __init__(self, device, channels, n_query,n_his, num_nodes, num_modals):
        super(MDA, self).__init__()
        self.channels = channels
        self.n_query = n_query
        self.c = channels
        self.t = n_his
        self.n = num_nodes
        self.m = num_modals
        self.att = nn.Conv2d(in_channels = self.channels, out_channels = self.n_query, kernel_size = (1,1))
        self.agg = nn.AvgPool2d(kernel_size=(self.n_query, 1), stride=1)
        # MoST Embedding
        self.temporal_embedding = nn.Parameter(torch.randn(channels, n_his), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.temporal_embedding)
        self.spatial_embedding = nn.Parameter(torch.randn(channels, num_nodes), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.spatial_embedding)
        self.modality_embedding = nn.Parameter(torch.randn(channels, num_modals), requires_grad=True).to(device)
        nn.init.xavier_normal_(self.modality_embedding)
        # Linear proj.
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels = 2*channels, out_channels = channels, kernel_size = (1,1,1)),
            nn.ReLU())
        
    def get_moste(self):
        """Generate the MoST Embedding E."""
        moste = self.temporal_embedding.reshape(1,self.c,1,1,self.t) + self.spatial_embedding.reshape(1,self.c,1,self.n,1) + self.modality_embedding.reshape(1,self.c,self.m,1,1)
        return moste
    
    def augmentation(self, x, sim, percent=0.1):
        """Generate the modality-aware augumentation."""
        x = x.permute(0,3,2,1,4)
        b,n,m = sim.shape
        mask_num = int(b * n * m * percent)        
        aug_x = copy.deepcopy(x)

        mask_prob = (1. - sim.reshape(-1)).numpy()              
        mask_prob /= mask_prob.sum()
        
        if np.logical_or.reduce(np.isnan(mask_prob)):
            raise ValueError("probabilities contain a value that is not a number")
        
        x, y, z = np.meshgrid(range(b), range(n), range(m), indexing='ij')
        mask_list = np.random.choice(b * n * m, size=mask_num, p=mask_prob)
        
        zeros = torch.zeros_like(aug_x[0, 0, 0])        
        aug_x[
            x.reshape(-1)[mask_list], 
            y.reshape(-1)[mask_list], 
            z.reshape(-1)[mask_list]] = zeros 
        return aug_x.permute(0,3,2,1,4)
    
    def forward(self, x, rep):
        b, c, m, n, _ = rep.shape
        rep = rep.permute(0,1,3,4,2).reshape(b,c,-1,m)
        # calculate the attention matrix A using key x
        A = self.att(rep)    
        A = torch.softmax(A, dim=-1)  
        # calculate the modality simlarity (prob)
        A = torch.einsum('bqlm->lbqm', A) 
        sim = self.agg(A).squeeze(2).permute(1,0,2) 
        # get the data augmentation
        aug_x = self.augmentation(x.detach(), sim.detach().cpu())
        # get the MoST embedding
        moste = self.get_moste().repeat(b,1,1,1,1)
        aug = self.proj(torch.cat((aug_x, moste), dim=1))
        return aug

##########################################################
## Global Self-Supervised Learning (GSSL)
##########################################################
LOG2PI = math.log(2 * math.pi)
class GSSL(nn.Module):
    def __init__(self, in_features, channels, num_comp):
        super(GSSL, self).__init__()
        self.in_features = in_features
        self.num_comp = num_comp
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.gamma = nn.Sequential(
            nn.Linear(in_features*channels, num_comp, bias=False),
            nn.Softmax(dim=-1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Conv1d(in_channels = in_features, out_channels = num_comp, kernel_size = 1)
        self.mu = nn.Conv1d(in_channels = in_features, out_channels = num_comp, kernel_size = 1)
        
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def get_GaussianPara(self, rep):
        '''
        :param rep: representation, [b,c,m,n,t]
        :return gammas: membership score, [b,k]
        :return sigmas, mus: Gaussian parameters (mean and variance), [b,k,c]
        '''
        b, c, m, n, _ = rep.shape
        gammas = self.gamma(rep.reshape(b,-1))  
        mus = self.mu(rep.permute(0,2,3,4,1).reshape(b,-1,c))
        sigmas = torch.exp(self.sigma(rep.permute(0,2,3,4,1).reshape(b,-1,c)))        
        return gammas.unsqueeze(1), mus.permute(0,2,1), sigmas.permute(0,2,1)
    
    def get_logPdf(self, rep, mus, sigmas):
        '''
        :param rep: representation, [b,c,m*n*t]
        :param sigmas, mus: Gaussian parameters, [b,c,k]
        return log_component_prob: log PDF, [b, m*n*t, k]
        '''
        h = rep.unsqueeze(-1)
        mus = mus.unsqueeze(2)
        sigmas = sigmas.unsqueeze(2)
        log_component_prob = -torch.log(sigmas) - 0.5 * LOG2PI - 0.5 * torch.pow((h - mus) / sigmas, 2)
        # torch.prod(log_component_prob, 1) may cause inf
        return self.l2norm(torch.prod(log_component_prob, 1))
        
    def forward(self, rep, rep_aug):
        b, c, m, n, _ = rep_aug.shape        
        rep = self.l2norm(rep)
        rep_aug = self.l2norm(rep_aug)
        gammas_aug, mus_aug, sigmas_aug = self.get_GaussianPara(rep_aug)
        # get log Pdf with the original representation H as a self-supervised signal
        log_component_prob_aug = self.get_logPdf(rep.reshape(b,c,-1), mus_aug, sigmas_aug)  
        log_prob_aug = log_component_prob_aug + torch.log(gammas_aug)
        # calculate loss
        loss = -torch.mean(torch.log(torch.sum(log_prob_aug.exp(), dim=-1)))
        return loss

##########################################################
## Modality Self-Supervised Learning (MSSL)
##########################################################
class MSSL(nn.Module):
    def __init__(self, channels, num_nodes, num_modals, device):
        super(MSSL, self).__init__()
        self.device = device
        self.flat_hidden = num_nodes * num_modals
        self.W1 = nn.Parameter(torch.FloatTensor(self.flat_hidden, channels))
        self.W2 = nn.Parameter(torch.FloatTensor(self.flat_hidden, channels)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))        
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Bilinear(channels, channels, 1)        
        self.logits_loss = nn.BCEWithLogitsLoss()
        
        for m in self.modules():
            self.weights_init(m)
        
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def fusion(self, rep, rep_aug):
        '''
        :param rep: original representation, [b,c,m,n,t]
        :param rep_aug: augmented representation, [b,c,m,n,t]
        :return h: fusion representation, [b,m,n,c]
        :return cm: unified modality representation, [b,m,c]
        '''
        b, c, m, n, _ = rep.shape
        rep = rep.permute(0,2,3,4,1).reshape(b,-1,c)
        rep_aug = rep_aug.permute(0,2,3,4,1).reshape(b,-1,c)
        h = (rep * self.W1 + rep_aug * self.W2).reshape(b,m,n,c) 
        # unified modality representation
        cm = torch.mean(h, dim=2)
        cm = self.sigmoid(cm)
        return h, cm
    
    def pn_sampling(self, h):
        '''
        :param h: fusion representation, [b,m,n,c]
        :return h: real hidden representation (w.r.t g), [b,m,n,c]
        :return shuf_h: fake hidden representation, [b,m,n,c]
        '''
        idx = torch.randperm(h.size(1))
        shuf_h = h[:,idx,:,:]
        return h, shuf_h
        
    def get_logits(self, cm, h_rl, h_fk):
        '''
        :param cm: unified modality representation, [b,m,c]
        :param h_rl: real hidden representation, [b,m,n,c]
        :param h_fk: fake hidden representation, [b,m,n,c]
        :return logits: scores, [b,m,n,2]
        '''        
        cm = torch.unsqueeze(cm, dim=2) 
        cm = cm.expand_as(h_rl).contiguous()  
        # score of real and fake
        sc_rl = self.net(h_rl.contiguous(), cm.contiguous())
        sc_fk = self.net(h_fk.contiguous(), cm.contiguous())
        logits = torch.cat((sc_rl, sc_fk), dim=-1)        
        return logits
    
    def cal_loss(self, logits):
        '''
        :param logits: scores, [b,m,n,2]
        :return loss: MSSL loss
        '''  
        b,m,n,_ = logits.shape
        l_rl = torch.ones(b,m,n,1)
        l_fk = torch.zeros(b,m,n,1)        
        l = torch.cat((l_rl, l_fk), dim=-1).to(self.device)
        loss = self.logits_loss(logits, l)
        return loss
    
    def forward(self, rep, rep_aug):
        # get the fusion representation h and the unified modality representation cm
        h, cm = self.fusion(rep, rep_aug)
        # positive and negative pair sampling
        h_rl, h_fk = self.pn_sampling(h)    
        # calculate loss
        logits = self.get_logits(cm, h_rl, h_fk)
        loss = self.cal_loss(logits)
        return loss
