import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import new_next_batch_all
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, dpp_loss
from util.dpp import map_matrix_index
from util.utils import EarlyStopManager

import wandb
import json

class Our(GraphRecommender):
    def __init__(self, conf, training_set, val_set, test_set, **kwargs):
        super(Our, self).__init__(conf, training_set, val_set, test_set, **kwargs)
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.dpp_rate = float(self.config['dppWeight'])
        
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.use_orinin = self.config['use_orinin']
        self.use_Gaussian = self.config['use_Gaussian']
        self.only_DPP = self.config['only_DPP']
        self.only_CL = self.config['only_CL']
        self.division_emb = self.config['division_emb']
        self.division_bt = self.config['division_bt']
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        
        self.sample_map = {"kk": new_next_batch_all}
        self.kwargs = {}
        self.sample_mode = self.config['sample_mode']
        self.sample_function = self.sample_map[self.sample_mode]
        if self.config['sample_mode'] == 'knk':
            self.kwargs['neg_pos_ratio'] = float(self.config['neg_pos_ratio'])
        self.kwargs['batch_size'] = self.batch_size

        self.max_metric = -1.0
        self.esm = EarlyStopManager(self.config)
        self.best_epoch = 0
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.init(project=self.config['project'], entity=self.config['entity'], name=self.config['wandb_name'])

    # 按用户采样 train
    def train(self):
        model = self.model.cuda()
        batch_size_map_n = {32: 160, 50:120, 64: 80, 128:40, 256:20, 512:10, 1024:5, 2048: 3}
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.sample_function(self.data, **self.kwargs)):
                user_idx, item_idx, u_map_pos, u_map_neg, user_idx_dict, item_idx_dict, simgcl_user_idx, simgcl_pos_idx, simgcl_neg_idx = batch
                all_user_emb, all_item_emb = model()
                dpp_user_emb, dpp_item_emb = all_user_emb[user_idx], all_item_emb[item_idx]
                u_map_pos_mindex, u_map_neg_mindex = map_matrix_index(user_idx_dict, item_idx_dict, u_map_pos, u_map_neg)
                
                simgcl_user_emb, simgcl_pos_item_emb, simgcl_neg_item_emb = all_user_emb[simgcl_user_idx], all_item_emb[simgcl_pos_idx],all_item_emb[simgcl_neg_idx]

                rec_dpp_loss = self.dpp_rate * dpp_loss(u_map_pos_mindex, u_map_neg_mindex, dpp_user_emb, dpp_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([simgcl_user_idx, simgcl_pos_idx])
                reg_loss = l2_reg_loss(self.reg, simgcl_user_emb, simgcl_pos_item_emb)

                batch_loss =  cl_loss + rec_dpp_loss + reg_loss

                optimizer.zero_grad()
                batch_loss.backward()

                optimizer.step()
                if n % batch_size_map_n[self.batch_size]==0:
                    print('training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item(), 'rec_dpp_loss', rec_dpp_loss.item(), 'batch_loss', batch_loss.item()) # 没有rec_loss
                    if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
                        wandb.log({"cl_loss": cl_loss.item(), "rec_dpp_loss": rec_dpp_loss.item(), "l2_reg_loss":reg_loss, "batch_loss": batch_loss.item()})
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            
            val_watch_metric = self.val_fast_evaluation(self.item_emb)
            print(f"epoch:{epoch + 1}   {self.config['val_watch_metric']}@{self.config['val_watch_metric_n']}: {val_watch_metric}")
            if val_watch_metric > self.max_metric:
                self.best_epoch = epoch + 1
                self.max_metric = val_watch_metric
                self.save()
            should_stop = self.esm.step(val_watch_metric)
            if should_stop:
                break
            
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.log({'epoch': self.best_epoch})
        self.update_best_emb(self.best_item_emb)

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_1, item_view_1 = self.model()
        user_view_2, item_view_2 = self.model(gaussian=True)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss
            

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, gaussian=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0) 
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if gaussian:
                tmp_noise = torch.ones_like(ego_embeddings).cuda()
                tmp_tmp_noise = tmp_noise * torch.clone(ego_embeddings).std()
                gaussian_noise = torch.normal(0, tmp_tmp_noise)
                ego_embeddings += gaussian_noise * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
