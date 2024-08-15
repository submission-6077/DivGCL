import math
import torch
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist
class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    @staticmethod
    def cal_tradeoff2(recall, NDCG, ILAD, ILMD,gini,coverage,entropy):
        accuracy = (recall + NDCG) / 2
        ILAD_ILMD = (ILAD + ILMD) / 2
        ILAD_gini = (ILAD + gini) / 2
        ILAD_coverage = (ILAD + coverage) / 2
        ILAD_entropy = (ILAD + entropy) / 2
        gini_coverage = (gini + coverage) / 2
        gini_entropy= (gini + entropy) / 2
        entropy_coverage = (entropy + coverage) / 2
        tradeoff = (2*accuracy*ILAD_ILMD) / (accuracy+ILAD_ILMD)
        t_ILAD_gini = (2*accuracy*ILAD_gini) / (accuracy+ILAD_gini)
        t_ILAD_coverage = (2*accuracy*ILAD_coverage) / (accuracy+ILAD_coverage)
        t_ILAD_entropy = (2*accuracy*ILAD_entropy) / (accuracy+ILAD_entropy)
        t_gini_coverage = (2*accuracy*gini_coverage) / (accuracy+gini_coverage)
        t_gini_entropy = (2*accuracy*gini_entropy) / (accuracy+gini_entropy)
        t_entropy_coverage = (2*accuracy*entropy_coverage) / (accuracy+entropy_coverage)

        return tradeoff, t_ILAD_gini, t_ILAD_coverage, t_ILAD_entropy, t_gini_coverage, t_gini_entropy, t_entropy_coverage
    def cal_tradeoff1(recall, NDCG, ILAD, ILMD):
        accuracy = (recall + NDCG) / 2
        ILAD_ILMD = (ILAD + ILMD) / 2
        tradeoff = (2*accuracy*ILAD_ILMD) / (accuracy+ILAD_ILMD)

        return tradeoff
    def cal_diversity(item_map_idx, res, item_emb):
        min_div_sum = 0
        mean_div_sum = 0
        user_size = len(res)
        for u in res:
            rec_item_list = res[u]
            item_idx_list = []
            for (item, score) in rec_item_list:
                item_idx_list.append(item_map_idx[item])

            rec_item_emb = item_emb[item_idx_list]
            div_distance = pdist(rec_item_emb.cpu().detach().numpy(), 'cosine')
            mean_div_sum += div_distance.mean()
            min_div_sum += div_distance.min()
        if min_div_sum < 0:
            min_div_sum = 0
        return mean_div_sum / user_size, min_div_sum / user_size
    @staticmethod
    def cal_category_num(res, category):
        user_item_category = []
        for u in res:
            rec_item_list = res[u]
            item_category = []
            for (item_idx, score) in rec_item_list:
                item_category.append(category[int(item_idx)])
            user_item_category.append(item_category)
        per_category_num_per_user = [np.unique(np.array(item_category), return_counts=True)[1] for item_category in user_item_category]
        
        return per_category_num_per_user

    @staticmethod
    def cal_gini(per_category_num_per_user: list):
        user_category_list = per_category_num_per_user
        user_size = len(user_category_list)

        gini_sum = 0
        for user_category in user_category_list:
            user_category = np.sort(user_category)
            n = user_category.size   
            cum_count = np.cumsum(user_category)  
            gini_sum += (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n

        gini = gini_sum / user_size
        return gini
    
    @staticmethod
    def cal_coverage(per_category_num_per_user: list):
        user_category_list = per_category_num_per_user
        user_size = len(user_category_list)

        user_category_sum = 0
        for user_category in user_category_list:
            user_category_sum += user_category.size
        coverage = user_category_sum / user_size

        return coverage

    @staticmethod
    def cal_entropy(per_category_num_per_user: list):
        user_category_list = per_category_num_per_user
        user_size = len(user_category_list)

        user_entropy_sum = 0
        for user_category in user_category_list:
            user_entropy_sum += entropy(user_category)
        entropy_result = user_entropy_sum / user_size

        return entropy_result

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)
def val_ranking_evaluation(item_map_idx, origin, res, N, item_emb, val_watch_metric, val_watch_metric_n, category=None):
    topN = max(N)
    if val_watch_metric_n <= topN:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:val_watch_metric_n]

        hits = Metric.hits(origin, predicted)
        if val_watch_metric == 'recall':
            return Metric.recall(hits, origin)
        elif val_watch_metric == 'precision':
            return Metric.precision(hits, val_watch_metric_n)
        elif val_watch_metric == 'ndcg':
            return Metric.NDCG(origin, predicted, val_watch_metric_n)
        elif val_watch_metric == 'hit_ratio':
            return Metric.hit_ratio(origin, hits)
        elif val_watch_metric == 'ilad':
            ILAD, ILMD = Metric.cal_diversity(item_map_idx, predicted, item_emb) 
            return ILAD
        elif val_watch_metric == 'ilmd':
            ILAD, ILMD = Metric.cal_diversity(item_map_idx, predicted, item_emb) 
            return ILMD   
def ranking_evaluation(item_map_idx, origin, res, N, item_emb, category=None):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')

        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        
        ILAD, ILMD = Metric.cal_diversity(item_map_idx, predicted, item_emb) 
        indicators.append('ILAD:' + str(ILAD) + '\n')
        indicators.append('ILMD:'  + str(ILMD) + '\n')

        tradeoff = Metric.cal_tradeoff1(recall,NDCG,ILAD,ILMD)
        indicators.append("tradeoff:" + str(tradeoff) + '\n')
        
        if category: 
            per_category_num_per_user = Metric.cal_category_num(predicted, category)
            gini = Metric.cal_gini(per_category_num_per_user)
            indicators.append('gini:' + str(gini) + '\n')
            
            coverage = Metric.cal_coverage(per_category_num_per_user)
            indicators.append('coverage:'  + str(coverage) + '\n')

            entropy_result = Metric.cal_entropy(per_category_num_per_user)
            indicators.append('entropy:'  + str(entropy_result) + '\n')
        
            tradeoff2, t_ILAD_gini, t_ILAD_coverage, t_ILAD_entropy, t_gini_coverage, t_gini_entropy, t_entropy_coverage = Metric.cal_tradeoff2(recall,NDCG,ILAD,ILMD,gini,coverage,entropy_result)
            indicators.append("tradeoff2:" + str(tradeoff2) + '\n')
            indicators.append("t_ILAD_gini:" + str(t_ILAD_gini) + '\n')
            indicators.append("t_ILAD_coverage:" + str(t_ILAD_coverage) + '\n')
            indicators.append("t_ILAD_entropy:" + str(t_ILAD_entropy) + '\n')
            indicators.append("t_gini_coverage:" + str(t_gini_coverage) + '\n')
            indicators.append("t_gini_entropy:" + str(t_gini_entropy) + '\n')
            indicators.append("t_gini_entropy:" + str(t_gini_entropy) + '\n')
            indicators.append("t_entropy_coverage:" + str(t_entropy_coverage) + '\n')

        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure