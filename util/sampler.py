from random import shuffle,randint,choice
import random
import torch
import numpy as np
from scipy.linalg import qr
import time
def DPPSamp(kernel_matrix, max_iter, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    c = np.zeros((max_iter, kernel_matrix.shape[0]))
    d = np.copy(np.diag(kernel_matrix.cpu().detach().numpy()))
    j = np.argmax(d)
    Yg = [j]
    iter = 0
    Z = list(range(kernel_matrix.shape[0]))
    while len(Yg) < max_iter:
        Z_Y = set(Z).difference(set(Yg))
        for i in Z_Y:
            if iter == 0:
                ei = kernel_matrix[j, i] / np.sqrt(d[j])
            else:
                ei = (kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
            c[iter, i] = ei
            d[i] = d[i] - ei * ei
        d[j] = 0
        j = np.argmax(d)
        if d[j] < epsilon:
            break
        Yg.append(j)
        iter += 1
    return Yg

def dpp_sample(pos_items: list, item_similarity_matrix: torch.tensor):
    items_num = item_similarity_matrix.shape[0]
    batch_all_items = range(items_num)
    neg_items = list(set(batch_all_items).difference(pos_items))
    item_map_dict = {}
    for index, neg_item in enumerate(neg_items):
        item_map_dict[index] = neg_item
    
    neg_item_similarity_matrix = item_similarity_matrix[neg_items][:, neg_items]

    sample_list = DPPSamp(neg_item_similarity_matrix, len(pos_items))

    sample_result = []
    for i in sample_list:
        sample_result.append(item_map_dict[i])
    return sample_result

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx

def next_batch_pairwise_user(data, batch_size, n_negs=1):
    batch_id = 0
    training_set_u = data.training_set_u
    user_size = len(training_set_u)
    training_user = [user for user in training_set_u]
    shuffle(training_user)
    while batch_id < user_size:
        if batch_id + batch_size <= user_size:
            users = [training_user[index] for index in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_user[index] for index in range(batch_id, user_size)]
            batch_id = user_size
        
        u_idx = []
        i_idx = []
        j_idx = []
        for user in users:
            pos_items_str = training_set_u[user]
            pos_items = [data.item[idx_str] for idx_str in pos_items_str] 
            i_idx.extend(pos_items)
            for i in range(len(pos_items)):
                u_idx.append(data.user[user])
                
            item_list = list(data.item.keys())
            for m in range(len(pos_items)):
                neg_item = choice(item_list)
                while neg_item in pos_items_str:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        yield u_idx, i_idx, j_idx

def new_next_batch_all(data, **kwargs):
    batch_size = kwargs['batch_size']
    batch_id = 0
    training_set_u = data.training_set_u
    user_size = len(training_set_u)
    training_user = [user for user in training_set_u]
    shuffle(training_user)
    while batch_id < user_size:
        user_idx = []
        item_idx = []
        u_map_pos = {}
        u_map_neg = {}
        item_idx_dict = {}
        user_idx_dict = {}

        simgcl_user_idx = []
        simgcl_pos_idx = []
        simgcl_neg_idx = []
        if batch_id + batch_size <= user_size:
            users = [training_user[index] for index in range(batch_id, batch_id + batch_size)]
            batch_id += batch_size
        else:
            users = [training_user[index] for index in range(batch_id, user_size)]
            batch_id = user_size
        
        for user in users:
            pos_items_dict = training_set_u[user]
            pos_items_idx = [data.item[pi] for pi in pos_items_dict]

            u_idx = data.user[user]
            user_idx.append(u_idx)

            u_map_pos[u_idx] = []
            u_map_neg[u_idx] = []

            u_map_pos[u_idx].extend(pos_items_idx)
            item_idx.extend(pos_items_idx)

            for i in range(len(pos_items_idx)):
                simgcl_user_idx.append(u_idx)
            simgcl_pos_idx.extend(pos_items_idx)

            all_items = list(data.item.keys())
            pos_items = list(pos_items_dict.keys())
            all_neg_items = list(set(all_items).difference(pos_items))
            neg_items = random.sample(all_neg_items, len(pos_items))

            for neg_item in neg_items:
                neg_item_idx = data.item[neg_item]
                u_map_neg[u_idx].append(neg_item_idx)
                item_idx.append(neg_item_idx)

                simgcl_neg_idx.append(neg_item_idx)
        

        user_idx = sorted(list(set(user_idx)))
        item_idx = sorted(list(set(item_idx)))
        for i, user in enumerate(user_idx):
            user_idx_dict[user] = i
        
        for i, item in enumerate(item_idx):
            item_idx_dict[item] = i
        
        yield user_idx, item_idx, u_map_pos, u_map_neg, user_idx_dict, item_idx_dict, simgcl_user_idx, simgcl_pos_idx, simgcl_neg_idx

    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y
