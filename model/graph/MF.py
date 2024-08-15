import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise, next_batch_pairwise_user
from util.loss_torch import bpr_loss,l2_reg_loss
from util.utils import EarlyStopManager
import wandb

class MF(GraphRecommender):
    def __init__(self, conf, training_set, val_set, test_set, **kwargs):
        super(MF, self).__init__(conf, training_set, val_set, test_set, **kwargs)
        self.model = Matrix_Factorization(self.data, self.emb_size)
        self.sample_map = {"data": next_batch_pairwise,
                        "user": next_batch_pairwise_user} 
        self.sample_mode = self.config['data_or_user']
        self.sample_function = self.sample_map[self.sample_mode]

        self.max_metric = -1.0
        self.esm = EarlyStopManager(self.config)
        self.best_epoch = 0
        
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.init(project=self.config['project'], entity=self.config['entity'], name=self.config['wandb_name'])

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(self.sample_function(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 20 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                    if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
                        wandb.log({"batch_loss": batch_loss.item()})
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            
            val_watch_metric = self.val_fast_evaluation(self.item_emb)
            print(f"epoch:{epoch + 1}   {self.config['val_watch_metric']}@{self.config['val_watch_metric_n']}: {val_watch_metric}")
            if val_watch_metric > self.max_metric:
                self.best_epoch = epoch + 1
                self.max_metric = val_watch_metric
                self.save()
            stop = self.esm.step(val_watch_metric)
            if stop:
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.log({'epoch': self.best_epoch})
        self.update_best_emb(self.best_item_emb)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']


