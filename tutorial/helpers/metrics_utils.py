class Metrics:
    def __init__(self):
        self.metrics = {}

    def update(self, model_name, metric_name, value):
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        if metric_name not in self.metrics[model_name]:
            self.metrics[model_name][metric_name] = []
        
        # if the incoming value is a list collapse them into one list
        if isinstance(value, list):
            self.metrics[model_name][metric_name].extend(value)
        else: 
            self.metrics[model_name][metric_name].append(value)
            
    def get_summary(self):
        raise NotImplementedError
    
    def reset(self, metric_name=None):
        if (metric_name is not None) and (metric_name in self.metrics.keys()):
            del self.metrics[metric_name]
        elif  (metric_name is not None) and (metric_name not in self.metrics.keys()):
            raise KeyError('metric_name \'{}\' not in metric keys'.format(metric_name))
        else:
            self.metrics.clear()

    def __getitem__(self, name):
        if name not in self.metrics:
                raise KeyError('metric \'{}\' does not exist'.format(name))
        return self.metrics[name]


class TrainingCallback:
    def __init__(self, metrics, model_name):
        self.metrics = metrics
        self.model_name = model_name

    def __call__(self, metric_name, value):
        self.metrics.update(self.model_name, metric_name, value)

class MetricTracker:
    def __init__(self):
        self.metrics = {}
        
    def track(self, name, value, epoch=None, batch=None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((epoch, batch, value))
    
    def get_metric(self, name):
        return self.metrics.get(name, None)
    
    def reset(self):
        self.metrics = {}

    def get_epoch_recon_accuracy(self, name, epoch, total_count):
        ''' Calculate the accuracy of the reconstructions agains the actuals '''
        metric_data = self.get_metric(name)
        if metric_data is None:
            raise ValueError('Metric \'{}\' does not exist.'.format(name))
        
        correct_count = []
        for epoch_i, batch_i, value_i in self.get_metric(name):
            if epoch_i == epoch:
                correct_count.append(value_i)

        assert  len(correct_count) != 0, 'there was not data for metric \'{}\' and epoch \'{}\''.format(name, epoch)

        return 100.0 * (sum(correct_count) / total_count) if total_count > 0 else 0.0

        # accuracy = 100.0 * (correct_count / total_count) if total_count > 0 else 0.0
        # self.track(metric_name, accuracy, epoch, batch)
    
    def get_epoch_average(self, name):
        ''' Calculate the average of each metric across batches for each epoch '''
        metric_data = self.get_metric(name)
        if metric_data is None:
            raise ValueError('Metric \'{}\' does not exist.'.format(name))
        
        epoch_data = {}
        for epoch, batch, value in self.get_metric(name):
            if epoch not in epoch_data:
                epoch_data[epoch] = []
            epoch_data[epoch].append(value)
        return [sum(epoch_data[epoch]) / len(epoch_data[epoch]) for epoch in sorted(epoch_data.keys())]

    def get_episode_total_reward(self, name):
        ''' get the total reward per episode '''
        metric_data = self.get_metric(name)
        if metric_data is None:
            raise ValueError('Metric \'{}\' does not exist.'.format(name))
        
        episode_data = {}
        for episode, batch, value in self.get_metric(name):
            if episode not in episode_data:
                episode_data[episode] = []
            episode_data[episode].append(value)
        return [sum(episode_data[episode]) for episode in sorted(episode_data.keys())]

