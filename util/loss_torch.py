import random
import torch
import torch.nn.functional as F
import math
from util.sampler import dpp_sample, DPPSamp
from scipy.linalg import qr
import numpy as np
import time
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = F.relu(neg_score+1-pos_score)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p+kl_q)

def dpp_loss(u_map_pos_i, u_map_neg_i, user_emb, item_emb):
    item_emb_norm = F.normalize(item_emb, dim=1)


    predict_score = torch.mm(user_emb, item_emb.t())
    user_size = len(u_map_pos_i)

    loss = torch.tensor(0)

    for u in u_map_pos_i:
        pos_items = u_map_pos_i[u]
        neg_items = u_map_neg_i[u]
        
        eps = 1e-5
        Tu_pos_score = predict_score[u][pos_items]
        Tu_neg_score = predict_score[u][neg_items] 
       
        loss1 = torch.sum(torch.log(torch.pow(Tu_pos_score, 2)))

        singularvals2 = torch.linalg.svdvals(item_emb_norm[pos_items])
        loss2 = singularvals2[singularvals2 > eps].log().sum().mul(2)

        loss3 = torch.sum(torch.log(torch.pow(Tu_neg_score, 2)))
        
        singularvals4 = torch.linalg.svdvals(item_emb_norm[neg_items])
        loss4 = singularvals4[singularvals4 > eps].log().sum().mul(2)

        pos_loss = loss1 + loss2
        neg_loss = loss3 - loss4
        each_dpp_loss = pos_loss / neg_loss
        loss = loss + each_dpp_loss
        
    return loss/user_size