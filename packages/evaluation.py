import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from packages.attacker import _Attacker, _ModelMixin
from packages.myDatasetsLoader import NIPS2017AdversaryCompetitionDataset
from packages.myVisualizeShow import single_batch_visualize, signle_batch_adversary, dataloader_acc
from packages.myDecorator import cuda_free_cache

class _Evaluation(_ModelMixin):

    def __init__(self, model=None, device='cpu', batch_size=50):
        self.device = torch.device(device)
        self.model = model
        self.batch_size = batch_size
        self._cleanImages_accuracy()
        self.limited_epsilons = [4, 8, 12, 16]

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def dataloader(self):
        return self._dataloader

    @property
    def acc_cleanImages(self):
        return self._acc_cleanImages

    @batch_size.setter
    def batch_size(self, value):
        if value < 1 or value > 100:
            raise ValueError('batch_size must be between 1~100')
        self._batch_size = value
        self._dataloader = DataLoader(NIPS2017AdversaryCompetitionDataset(root_dir='./Data/nips-2017-adversarial-learning-development-set', transform=transforms.Compose([
            transforms.ToTensor(),])), batch_size=self.batch_size, shuffle=True)
    
    @property
    def limited_epsilons(self):
        return self._limited_epsilons

    @limited_epsilons.setter
    def limited_epsilons(self, value):
        for epsilon in value:
            if not isinstance(epsilon, int):
                raise ValueError('epsilon must be int')
        self._limited_epsilons = value

    @property
    def evaluation(self):
        return self._evaluation

    @evaluation.setter
    def evaluation(self, value):
        self._evaluation = value

    @cuda_free_cache
    def cleanImages_visualize(self):
        single_batch_visualize(self.model, self.device, self.dataloader)

    @cuda_free_cache
    def _cleanImages_accuracy(self):
        self._acc_cleanImages, time = dataloader_acc(self.model, self.device, self.dataloader)
        self._evaluation = pd.DataFrame({'Name': 'clean_images',
                                        'Epsilon': pd.Series(0, index=list(range(1)), dtype='int'), 
                                        'Accuracy': self.acc_cleanImages, 
                                        'Time': time})

    def evaluation_visualize(self, names):
        plt.figure(figsize=(20, 10))
        plt.plot(self.limited_epsilons, np.ones_like(self.limited_epsilons)*self.acc_cleanImages, label='clean_images')
        for name in names:
            df = self.evaluation[self.evaluation.Name == name]
            plt.plot(df.Epsilon, df.Accuracy, label=name)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, max(self.limited_epsilons), step=1))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

class _Evaluation_Attack(_Evaluation):

    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)
        self._attackers = []

    @property
    def attackers(self):
        return self._attackers

    @property
    def attacker_names(self):
        return [x.__class__.__name__ for x in self.attackers]

    def _isAttacker(self, attacker):
        if not isinstance(attacker, _Attacker):
            raise ValueError('attacker must be an attacker class')
        if attacker in self.attackers:
            raise ValueError('attacker already exist')

    def add_attacker(self, attackers):
        if isinstance(attackers, (list, tuple)):
            for attacker in attackers:
                self._isAttacker(attacker)
                self.attackers.append(attacker)
        else:
            self._isAttacker(attackers)
            self.attackers.append(attackers)

    @cuda_free_cache
    def _attacker_visualize(self, attacker):
        signle_batch_adversary(self.model, self.device, self.dataloader, attacker)

    @cuda_free_cache
    def attackers_visualize(self):
        random_epsilon = random.choice(self.limited_epsilons)
        print(f'\n Maxiumum allowed perturbation is {random_epsilon}/255')
        for attacker in self.attackers:
            attacker.epsilon = random_epsilon
            self._attacker_visualize(attacker)

class Evaluation_Attack_NoTarget(_Evaluation_Attack):
    
    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)

    @cuda_free_cache
    def _attack_once(self, epsilon, attacker):
        attacker.epsilon = epsilon
        return dataloader_acc(self.model, self.device, self.dataloader, attacker)

    @cuda_free_cache
    def noTarget_evaluation(self):
        self.evaluation = self.evaluation[self.evaluation.Name == 'clean_images']       
        for attacker in self.attackers:
            accuracies = []
            times = []
            for epsilon in self.limited_epsilons:
                accuracy, time = self._attack_once(epsilon, attacker)
                accuracies.append(accuracy)
                times.append(time)
            df = pd.DataFrame({'Name': attacker.__class__.__name__,
                                'Epsilon': pd.Series(self.limited_epsilons), 
                                'Accuracy': pd.Series(accuracies),
                                'Time': times})
            self.evaluation = pd.concat([self.evaluation, df], ignore_index=True)
        print(self.evaluation)
        self.evaluation_visualize(self.attacker_names)