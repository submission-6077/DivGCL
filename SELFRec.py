from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.kwargs = {}
        if config['model.type'] == 'sequential':
            self.training_data, self.test_data = FileIO.load_data_set(config['sequence.data'], config['model.type'])
        else:
            self.training_data = FileIO.load_data_set(config['dataset'], config['training.set'], config['model.type'])
            self.val_data = FileIO.load_data_set(config['dataset'], config['val.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(config['dataset'], config['test.set'], config['model.type'])
            
            if config.contain('category.set'):
                self.kwargs['category'] = FileIO.load_category_set(config['dataset'],config['category.set'], config['model.type'])
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config, self.training_data, self.val_data, self.test_data,**self.kwargs)'
        eval(recommender).execute()
        