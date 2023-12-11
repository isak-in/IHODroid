import configparser
import os
import numpy as np
import torch as th
from .utils.activation import act_dict


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0:
            if th.cuda.is_available():
                self.device = th.device('cuda', int(gpu))
            else:
                raise ValueError("cuda is not available, please set 'gpu' -1")

        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.task = task
        self.model = model
        self.dataset = dataset
        self.output_dir = './openhgnn/output/{}'.format(self.model)
        self.optimizer = 'Adam'


        if model == 'Ho2vec':
            self.lr = conf.getfloat("Ho2vec", "learning_rate")
            self.weight_decay = conf.getfloat("Ho2vec", "weight_decay")
            self.seed = conf.getint("Ho2vec", "seed")
            self.dropout = conf.getfloat("Ho2vec", "dropout")
            self.n_layers = conf.getint("Ho2vec", "n_layers")
            self.in_dim = conf.getint('Ho2vec', 'in_dim')
            self.hidden_dim = conf.getint('Ho2vec', 'hidden_dim')
            self.out_dim = conf.getint('Ho2vec', 'out_dim')
            self.patience = conf.getint('Ho2vec', 'patience')
            self.max_epoch = conf.getint('Ho2vec', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("Ho2vec", "mini_batch_flag")


            
    def __repr__(self):
        return '[Config Info]\tModel: {},\tTask: {},\tDataset: {}'.format(self.model, self.task, self.dataset)