from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation, val_ranking_evaluation
import sys
import wandb
import re
import torch
from util.Category_DPP import DPP
import torch.nn.functional as F

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, val_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, val_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, val_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.isDPP = bool(self.config['is_DPP'])
        self.max_N = max(self.topN)
        self.val_watch_metric = self.config['val_watch_metric']
        self.val_watch_metric_n = int(self.config['val_watch_metric_n'])

        self.best_epoch = 0
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.init(project=self.config['project'], entity=self.config['entity'], name=self.config['wandb_name'])
    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Validate Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.val_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self, data_type):
        
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        rec_list = {}
        if data_type == 'val':
            eva_set = self.data.val_set
        elif data_type == 'test':
            eva_set = self.data.test_set

        user_count = len(eva_set)
        for i, user in enumerate(eva_set):
            candidates = self.predict(user)
            
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list, data_type):
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
    
        out_dir = self.output['-dir']
        
        if self.config.contain('dppWeight'):
            file_name = self.config['model.name'] + '@' + self.config['dataset'] + '@' + self.config['batch_size'] + '@' + self.config['dppWeight'] + '@' + current_time + '.txt'
        else:
            file_name = self.config['model.name'] + '@' + self.config['dataset'] + '@' + self.config['batch_size'] + '@' + '@' + current_time + '.txt'
        if data_type == 'val':
            eva_set = self.data.val_set
            
        elif data_type == 'test':
            eva_set = self.data.test_set
        print(self.category)
        
        if self.category:
            self.result = ranking_evaluation(self.data.item, eva_set, rec_list, self.topN, self.item_emb, self.category)
        else:
            self.result = ranking_evaluation(self.data.item, eva_set, rec_list, self.topN, self.item_emb)
        # submit to wandb
        top_pattern = re.compile(r'Top ([0-9]{1,})')
        wandb_evaluation = {}
        n = 0
        for m in self.result:
            if re.findall(top_pattern, m):
                n = re.findall(top_pattern, m)[0]
                continue
            k, v = m.strip().split(':')
            wandb_evaluation[k + '@' + str(n)] = float(v)
        if self.config.contain('is_wandb') and self.config['is_wandb']=='True':
            wandb.log(wandb_evaluation)

        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))
    
    def update_best_emb(self, item_emb):
        self.item_emb = item_emb
    
    def val_fast_evaluation(self, item_emb):
        data_type = 'val'
        print('evaluating the model...')
        rec_list = self.test(data_type)
        return val_ranking_evaluation(self.data.item, self.data.val_set, rec_list, self.topN, item_emb, self.val_watch_metric, self.val_watch_metric_n, self.category)
    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        data_type = 'test'
        rec_list = self.test(data_type)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''

        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure