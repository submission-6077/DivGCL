class EarlyStopManager(object):
    def __init__(self, config):

        # self.name = flags_obj.name + '_esm'
        # self.min_lr = flags_obj.min_lr
        self.es_patience = int(config['es_patience'])
        self.count = 0
        self.max_metric = -1.0

    def step(self, metric):
        if metric > self.max_metric:
            self.max_metric = metric
            self.count = 0
            return False
        else:
            self.count = self.count + 1
            if self.count > self.es_patience:
                return True
            return False