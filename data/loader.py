import os.path
from os import remove
from re import split


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    def read_cf_amazon(dataset,file_name):
        data = []
        with open(file_name) as f:
            for line in f:
                if dataset == 'amazon' or dataset == 'ali' or dataset == 'ml-1m':
                    items = split('\t', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = '1'
                elif dataset == 'douban-book' or dataset == 'Anime_Me' or dataset == 'API':
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = '1'
                elif dataset == 'Beauty' or dataset == 'Taobao' or dataset == 'msd' or dataset =='ml-1m_single_cate' or dataset == 'Anime' or dataset == 'Anime_Me2':
                    items = split(',', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = '1'
                else:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                data.append([user_id, item_id, float(weight)])
        return data
    @staticmethod
    def read_cf_aminer(dataset,file_name):
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for line in lines:
            tmps = line.strip()
            inters = [i for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id, float(1)])
        return list(inter_mat)

    @staticmethod
    def load_data_set(dataset, file, dtype):    
        if dtype == 'graph':
            if dataset == 'gowalla' or dataset == 'aminer':
                data = FileIO.read_cf_aminer(dataset, file)
            else:
                data = FileIO.read_cf_amazon(dataset, file)

        if dtype == 'sequential':
            training_data, test_data = [], []
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    user_id = items[0]
                    seq = items[1].strip().split()
                    training_data.append(seq[:-1])
                    test_data.append(seq[-1])
                data = (training_data, test_data)
        
        return data

    @staticmethod
    def load_category_set(dataset, file, dtype):
        category_dict = {}
        if dtype == 'graph':
            with open(file) as f:
                for line in f:
                    if dataset == 'msd' or dataset == 'ml-1m_single_cate' or dataset=='Anime'  or dataset=='Anime_Me' or dataset=='Anime_Me2' or dataset=='Beauty':
                        items = split(',', line.strip())
                        item_id = int(items[0])
                        category = int(items[1])
                        category_dict[item_id] = category
                    elif dataset == 'Anime_Me2':
                        items = split(' ', line.strip())
                        item_id = int(items[0])
                        category = int(items[1])
                        category_dict[item_id] = category
        return category_dict

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data
