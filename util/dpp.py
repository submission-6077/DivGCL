import random
import torch
def map_matrix_index(user_idx_dict, item_idx_dict, u_map_pos_i, u_map_neg_i):
    u_map_pos_mindex = {}
    u_map_neg_mindex = {}
    for user in u_map_pos_i:
        pos_items = u_map_pos_i[user]
        neg_items = u_map_neg_i[user]
        u = user_idx_dict[user]
        u_map_pos_mindex[u] = []
        u_map_neg_mindex[u] = []
        for pos_item in pos_items:
            pos_item_mindex = item_idx_dict[pos_item]
            u_map_pos_mindex[u].append(pos_item_mindex)
        for neg_item in neg_items:
            neg_item_mindex = item_idx_dict[neg_item]
            u_map_neg_mindex[u].append(neg_item_mindex)
    return u_map_pos_mindex, u_map_neg_mindex

def get_user_pos_neg_idx(u_map_pos_i, u_map_neg_i):
    user_idx = []
    pos_idx = []
    neg_idx = []
    for u in u_map_pos_i:
        pos_items = u_map_pos_i[u]
        neg_items = u_map_neg_i[u]
        for pos_item in pos_items:
            user_idx.append(u)
            pos_idx.append(pos_item)
        for neg_item in neg_items:
            neg_idx.append(neg_item)
    
    return user_idx, pos_idx, neg_idx
