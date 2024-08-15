from data.data import Data
from util.conf import OptionConf
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time


class Recommender(object):
    def __init__(self, conf, training_set, val_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, val_set, test_set)
        self.model_name = self.config['model.name']
        self.ranking = OptionConf(self.config['item.ranking'])
        self.emb_size = int(self.config['embbedding.size'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.lRate = float(self.config['learnRate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = OptionConf(self.config['output.setup'])
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, self.model_name + ' ' + current_time)
        self.result = []
        self.recOutput = []
        if 'category' in kwargs:
            self.category = kwargs['category']
        else:
            self.category = None

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        for k in self.config.config:
            self.model_log.add(k + '=' + self.config[k])

    def print_model_info(self):
        print('Model:', self.config['model.name'])
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:',  self.reg)
        print('gpu:', self.config['gpu'])
        if self.config.contain('sample_mode'):
            sample_mode = self.config['sample_mode']
            print(f'sample mode:{sample_mode}')
        if self.config.contain('dppWeight'):
            dppWeight = self.config['dppWeight']
            print(f'DppWeight:{dppWeight}')
        
        
        parStr = ''
        if self.config.contain(self.config['model.name']):
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        # print('Evaluating...')
        rec_list = self.test('test')     
        self.evaluate(rec_list, 'test')
