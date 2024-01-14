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
## Adaptive Augmentation
########################################    
class adaptiveAugmentation(nn.Module):
    def __init__(self, device, channels, n_query,n_his, num_nodes, num_modals):
        super(adaptiveAugmentation, self).__init__()
        self.channels = channels
        self.n_query = n_query
        self.c = channels
        self.t = n_his
        self.n = num_nodes
        self.m = num_modals
        self.att = nn.Conv2d(in_channels = self.channels, out_channels = self.n_query, kernel_size = (1,1))
        self.agg = nn.AvgPool2d(kernel_size=(self.n_query, 1), stride=1)
        # STM Embedding
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
        
    def get_stme(self):
        stme = self.temporal_embedding.reshape(1,self.c,1,1,self.t) + self.spatial_embedding.reshape(1,self.c,1,self.n,1) + self.modality_embedding.reshape(1,self.c,self.m,1,1)
        return stme
    
    def augmentation(self, x, sim, percent=0.1):
        """Generate the data augumentation from source attribute perspective."""
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
        # calculate the source simlarity (prob)
        A = torch.einsum('bqlm->lbqm', A) 
        sim = self.agg(A).squeeze(2).permute(1,0,2) 
        # get the data augmentation
        aug_x = self.augmentation(x.detach(), sim.detach().cpu())
        stme = self.get_stme().repeat(b,1,1,1,1)
        aug = self.proj(torch.cat((aug_x, stme), dim=1))
        return aug

##########################################################
## Global Heterogeneity Extractor (GHE)
##########################################################
LOG2PI = math.log(2 * math.pi)
class GHE(nn.Module):
    def __init__(self, in_features, channels, num_comp):
        super(GHE, self).__init__()
        self.in_features = in_features
        self.num_comp = num_comp
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.alpha = nn.Sequential(
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
        :return alphas: priori probability, [b,k]
        :return sigmas, mus: Gaussian parameters, [b,k,c]
        '''
        b, c, m, n, _ = rep.shape
        alphas = self.alpha(rep.reshape(b,-1))  
        mus = self.mu(rep.permute(0,2,3,4,1).reshape(b,-1,c))
        sigmas = torch.exp(self.sigma(rep.permute(0,2,3,4,1).reshape(b,-1,c)))        
        return alphas.unsqueeze(1), mus.permute(0,2,1), sigmas.permute(0,2,1)
    
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
        
    def get_cluAssignment(self, log_prob):
        '''
        :param log_prob: log PDF, [b, m*n*t, k]
        :return weightedlogPdf: [b, m*n*t, k]
        :return clusters: [b, m*n*t]
        '''
        log_sum = torch.log(torch.sum(log_prob.exp(), dim=-1, keepdim=True))        
        weightedlogPdf = log_prob - log_sum
        clusters = torch.argmax(weightedlogPdf,dim=-1).float()                 
        return clusters, weightedlogPdf
        
    def forward(self, rep, rep_aug):
        """Compute the contrastive loss of batched data."""
        b, c, m, n, _ = rep_aug.shape        
        rep = self.l2norm(rep)
        rep_aug = self.l2norm(rep_aug)
        alphas_aug, mus_aug, sigmas_aug = self.get_GaussianPara(rep_aug)
        # get log Pdf   
        log_component_prob_aug = self.get_logPdf(rep.reshape(b,c,-1), mus_aug, sigmas_aug)  
        log_prob_aug = log_component_prob_aug + torch.log(alphas_aug)
        # calculate loss
        loss = -torch.mean(torch.log(torch.sum(log_prob_aug.exp(), dim=-1)))
        return loss

##########################################################
## Cross-Modality Contrastive Learning (CMCL)
##########################################################
class CMCL(nn.Module):
    def __init__(self, channels, num_nodes, num_modals, device):
        super(CMCL, self).__init__()
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
        :param rep: representation form original input, [b,c,m,n,t]
        :param rep_aug: representation form augmentation, [b,c,m,n,t]
        :return h: fusion representation, [b,m,n,c]
        :return gs: global representation, [b,m,c]
        '''
        b, c, m, n, _ = rep.shape
        rep = rep.permute(0,2,3,4,1).reshape(b,-1,c)
        rep_aug = rep_aug.permute(0,2,3,4,1).reshape(b,-1,c)
        h = (rep * self.W1 + rep_aug * self.W2).reshape(b,m,n,c) 
        # global source representation
        gs = torch.mean(h, dim=2)
        gs = self.sigmoid(gs)
        return h, gs
    
    def pn_sampling(self, h):
        '''
        :param h: fusion representation, [b,m,n,c]
        :return h: real hidden representation (w.r.t g), [b,m,n,c]
        :return shuf_h: fake hidden representation, [b,m,n,c]
        '''
        idx = torch.randperm(h.size(1))
        shuf_h = h[:,idx,:,:]
        return h, shuf_h
        
    def discriminator(self, gs, h_rl, h_fk):
        '''
        :param gs: global representation, [b,n,c]
        :param h_rl: real hidden representation (w.r.t g), [b,m,n,c]
        :param h_fk: fake hidden representation, [b,m,n,c]
        :return logits: prediction scores, [b,m,n,2]
        '''        
        gs = torch.unsqueeze(gs, dim=2) 
        gs = gs.expand_as(h_rl).contiguous()  
        # score of real and fake
        sc_rl = self.net(h_rl.contiguous(), gs.contiguous())
        sc_fk = self.net(h_fk.contiguous(), gs.contiguous())
        logits = torch.cat((sc_rl, sc_fk), dim=-1)        
        return logits
    
    def cal_loss(self, logits):
        '''
        :param logits: prediction scores, [b,m,n,2]
        :return loss: contrastive loss
        '''  
        b,s,n,_ = logits.shape
        l_rl = torch.ones(b,m,n,1)
        l_fk = torch.zeros(b,m,n,1)        
        l = torch.cat((l_rl, l_fk), dim=-1).to(self.device)
        loss = self.logits_loss(logits, l)
        return loss
    
    def forward(self, rep, rep_aug):
        # get fusion representation h and globle representation gs
        h, gs = self.fusion(rep, rep_aug)
        # positive and negative pair sampling
        h_rl, h_fk = self.pn_sampling(h)    
        # calculate loss
        logits = self.discriminator(gs, h_rl, h_fk)
        loss = self.cal_loss(logits)
        return loss
