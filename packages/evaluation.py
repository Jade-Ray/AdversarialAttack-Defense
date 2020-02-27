import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from packages.attacker import _Attacker, _ModelMixin
from packages.myDatasetsLoader import NIPS2017AdversaryCompetitionDataset
from packages.myVisualizeShow import single_batch_visualize, signle_batch_adversary, dataloader_acc

class _Evaluation(_ModelMixin):

    def __init__(self, model=None, device='cpu', batch_size=50):
        self.device = torch.device(device)
        self.model = model
        self.batch_size = batch_size
        self._cleanImages_accuracy()

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

    def cleanImages_visualize(self):
        single_batch_visualize(self.model, self.device, self.dataloader)

    def _cleanImages_accuracy(self):
        self._acc_cleanImages = dataloader_acc(self.model, self.device, self.dataloader)

class Epsilon_Eval_Mixin(object):
    
    def _visualize(self, epsilons, results):
        plt.figure(figsize=(20, 10))
        for rusult in results:
            plt.plot(epsilons, rusult['accuracies'], label=rusult['name'])
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, max(epsilons), step=1))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def _calculate_accurasies(self, model, device, dataloader, epsilons, attacker):
        accuracies = []
        for epsilon in epsilons:
            attacker.epsilon = int(epsilon)
            acc = dataloader_acc(model, device, dataloader, attacker)
            accuracies.append(acc)
        return accuracies

class Evaluation_NoTarget(_Evaluation, Epsilon_Eval_Mixin):
    
    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)
        self._attackers = []

    @property
    def attackers(self):
        return self._attackers

    def add_attacker(self, attackers):
        if isinstance(attackers, (list, tuple)):
            for attacker in attackers:
                if not isinstance(attacker, _Attacker):
                    raise ValueError('attacker must be an attacker class')
                self._attackers.append(attacker)
        else:
            if not isinstance(attacker, _Attacker):
                raise ValueError('attacker must be an attacker class')
            self._attackers.append(attacker) 

    def epsilon_evaluation(self, epsilons):
        results = []
        results.append({'name': 'clean Images', 'accuracies': np.ones_like(epsilons)*self.acc_cleanImages})
        for attacker in self.attackers:
            accuracies = self._calculate_accurasies(self.model, self.device, self.dataloader, epsilons, attacker)
            results.append({'name': attacker.__class__.__name__, 'accuracies': accuracies})
        self._visualize(epsilons, results)
