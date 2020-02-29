import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

from packages.attacker import _Attacker, _ModelMixin
from packages.myDatasetsLoader import NIPS2017AdversaryCompetitionDataset
from packages.myVisualizeShow import visualize_adversary_batch, visualize_batch, visualize_targetedAdversary_batch, visualize_classifier_distribution
from packages.myDecorator import cuda_free_cache

class _Evaluation(_ModelMixin):

    def __init__(self, model=None, device='cpu', batch_size=50):
        self.device = torch.device(device)
        self.model = model
        self.batch_size = batch_size
        self.topk = (1, 5)
        self._cleanImages_accuracy()
        self.limited_epsilons = [4, 8, 12, 16]
        self._attackers_noTar = []
        self._attackers_tar = []
        self._defensors =[]

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def acc_cleanImages(self):
        return self._acc_cleanImages

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value):
        if value < 1 or value > 100:
            raise ValueError('batch_size must be between 1~100')
        self._batch_size = value
        self._load_dataloader()
    
    @property
    def topk(self):
        return self._topk
    
    @topk.setter
    def topk(self, value):
        if not isinstance(value, tuple):
            raise ValueError('topk must be a tuple')
        self._topk = value

    @property
    def batches(self):
        return self._batches

    @property
    def class_names(self):
        return self._class_names

    @property
    def data_num(self):
        return self._data_num

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
    def evaluation_cleanImages(self):
        return self._evaluation_cleanImages

    @property
    def least_likely_target(self):
        return self._least_likely_target

    @property
    def most_likely_target(self):
        return self._most_likely_target

    def _load_dataloader(self):
        self._dataloader = DataLoader(NIPS2017AdversaryCompetitionDataset(root_dir='./Data/nips-2017-adversarial-learning-development-set', transform=transforms.Compose([
            transforms.ToTensor(),])), batch_size=self.batch_size, shuffle=True)
        self._class_names = self.dataloader.dataset.class_names
        self._data_num = len(self.dataloader.dataset)
        self._batches = []
        for batch in self.dataloader:
            self._batches.append(batch)

    def _randomly_choosen_epsilon(self):
        random_epsilon = random.choice(self.limited_epsilons)
        print(f'\n Maxiumum allowed perturbation is {random_epsilon}/255')
        return random_epsilon

    def _randomly_choosen_target(self):
        random_target = random.choice(range(len(self.class_names)))
        print(f'\n Randomly Attack target is {random_target}({self.class_names[random_target]})')
        return random_target

    def _batch_correct(self, batch, correct, LTE=None, model=None):
        if not model:
            model = self.model
        maxk = max(self.topk)
        with torch.no_grad():
            inputs, targets = batch
            outputs = torch.softmax(model(inputs), dim=1)
            _, preds = outputs.topk(maxk, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(targets.view(1, -1).expand_as(preds))
            for i, k in enumerate(self.topk):
                correct[i] += corrects[:k].view(-1).float().sum(0).item()
            if LTE:
                for i, prob in enumerate([outputs[x, targets[x]] for x in range(outputs.size(0))]):
                    if prob.item() < LTE['min'][1]:
                        LTE['min'][0], LTE['min'][1] = targets[i].item(), prob.item()
                    if prob.item() > LTE['max'][1]:
                        LTE['max'][0], LTE['max'][1] = targets[i].item(), prob.item()
        return correct

    def _batch_parse(self, batch, targeted=False):
        inputs, targets = batch
        if isinstance(targeted, bool) and not targeted:
            pass
        elif isinstance(targeted, bool) and targeted:
            targets = torch.full_like(targets, self._randomly_choosen_target())
        elif isinstance(targeted, str) and targeted == 'most':
            targets = torch.full_like(targets, self.most_likely_target)
        elif isinstance(targeted, str) and targeted == 'least':
            targets = torch.full_like(targets, self.least_likely_target)
        elif isinstance(targeted, int) and targeted >= 0 and targeted < len(self.class_names):
            targets = torch.full_like(targets, targeted)
        else:
            raise ValueError('targeted must be bool or str(most or least) or int limited num_class')
        return (inputs, targets)

    def _batch_attack(self, batch, attacker, correct, time_elapsed, targeted=False):
        inputs, targets = self._batch_parse(batch, targeted)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        since = time.time()
        perturbed_inputs = attacker(inputs, targets)        
        correct = self._batch_correct((perturbed_inputs, targets), correct)
        time_elapsed += time.time() - since

        return (perturbed_inputs.detach().cpu(), targets.detach().cpu()), correct, time_elapsed

    def _batch_accuracy(self, batch, correct, time_elapsed, LTE=None, model=None):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        since = time.time()
        correct = self._batch_correct((inputs, targets), correct, LTE, model)
        time_elapsed += time.time() - since

        return correct, time_elapsed

    def _batches_accuracy(self, batches, LTE=None, model=None):
        time_elapsed = 0
        correct = np.zeros_like(self.topk)
        for batch in batches:
            correct, time_elapsed = self._batch_accuracy(batch, correct, time_elapsed, LTE, model)
        accuracy = correct / self.data_num
        return accuracy, correct, time_elapsed

    def _difference_maxNorm(self, cleanImgs, advImgs):
        difference = torch.sub(cleanImgs, advImgs)
        return torch.norm(difference, float('inf'), dim=(1, 2, 3)).numpy()

    def _create_dataframe(self, name, epsilon, accuracy, maxNorm, time, grade):
        return pd.DataFrame({'Name': name,
                            'Epsilon': epsilon, 
                            'Accuracy': accuracy,
                            'MaxNorm': maxNorm,
                            'Time': time, 
                            'Grade': grade,
                            'Topk': self.topk})

    @cuda_free_cache
    def _cleanImages_accuracy(self):
        likely_target_ex = {'min': [0, 1.], 'max': [0, 0.]}
        accuracy, correct, time_elapsed = self._batches_accuracy(self.batches, LTE=likely_target_ex)
        print(f'Least-Likely Class is {likely_target_ex["min"][0]}(prob = {likely_target_ex["min"][1]*100:.4f}%)')
        print(f'most-Likely Class is {likely_target_ex["max"][0]}(prob = {likely_target_ex["max"][1]*100:.4f}%)')
        self._least_likely_target = likely_target_ex['min'][0]
        self._most_likely_target = likely_target_ex['max'][0]
        self._acc_cleanImages = accuracy
        model_name = self.model.__class__.__name__
        self._evaluation_cleanImages = self._create_dataframe(model_name+'_clean_images', 0, accuracy, 0, time_elapsed, correct)

    @cuda_free_cache
    def _attack(self, attacker, epsilon, targeted=False):
        batches_adv = []
        max_norms = []
        correct = np.zeros_like(self.topk)
        time_elapsed = 0
        attacker.epsilon = epsilon
        for batch in self.batches:
            batch_adv, correct, time_elapsed = self._batch_attack(batch, attacker, correct, time_elapsed, targeted)
            batches_adv.append(batch_adv)
            max_norms.append(self._difference_maxNorm(batch[0], batch_adv[0]))
        accurary = correct / self.data_num
        maxNorm = np.max(max_norms) * 255
        evaluation_adv = self._create_dataframe(attacker.__class__.__name__, epsilon, accurary, maxNorm, time_elapsed, correct)
        
        print(f'\nAcctack model with {attacker.name} method')
        for i, k in enumerate(self.topk):
            print(f'Spend time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\tTop{k} Acc: {accurary[i]*100:.2f}%([{correct[i]}]/[{self.data_num}])')  

        return batches_adv, evaluation_adv

    @cuda_free_cache
    def _blackBox_model(self, model, batches_adv):
        model.to(self.device)
        model.eval()
        model_name = model.__class__.__name__
        accuracy, correct, time_elapsed = self._batches_accuracy(self.batches, model=model)
        df_cleanImages = self._create_dataframe(model_name+'_clean_images', 0, accuracy, 0, time_elapsed, correct)
        accuracy, correct, time_elapsed = self._batches_accuracy(batches_adv, model=model)
        df_adversary = self._create_dataframe(model_name+'_adversary', 0, accuracy, 0, time_elapsed, correct)
        return pd.concat([df_cleanImages, df_adversary], ignore_index=True)

    def _blackBox_ensemble(self, ensemble, attacker, epsilon, targeted=False, save=False):
        batches_adv, evaluation_adv = self._attack(attacker, epsilon, targeted)
        evaluation_balckBox = [self.evaluation_cleanImages, evaluation_adv]
        for model in ensemble:
            evaluation_balckBox.append(self._blackBox_model(model, batches_adv))
           
            accuracy_b, correct_b, _ = self._batches_accuracy(self.batches, model=model)
            accuracy_a, correct_a, _ = self._batches_accuracy(batches_adv, model=model)
            print('\nModel: {model.__class__.__name__}')           
            for i, k in enumerate(self.topk):
                print(f'before attack:\nTop{k} Acc: {accuracy_b[i]*100:.2f}%([{accuracy_b[i]}]/[{self.data_num}])')  
                print(f'after attack({attacker.__class__.__name__}):\nTop{k} Acc: {accuracy_a[i]*100:.2f}%([{accuracy_a[i]}]/[{self.data_num}])')
            
        evaluation_df = pd.concat(evaluation_balckBox, ignore_index=True)
        print(evaluation_df)
        if save:
            if targeted:
                evaluation_df.to_csv('attack_blackBox_targeted.csv')
            else:
                evaluation_df.to_csv('attack_blackBox_nonTargeted.csv')

    def visualize_cleanImages(self, model=None):
        if not model:
            model = self.model
        visualize_batch(model, self.device, self.batches[0], self.class_names)

    def visualize_distribution(self, model=None, batches=None):
        if not model:
            model = self.model
        if not batches:
            batches = self.batches
        visualize_classifier_distribution(model, self.device, batches, class_num = len(self.class_names))

class _NonTargetedAttackerMixin(object):

    @property
    def attackers_noTar(self):           
        return self._attackers_noTar

    def _isAttacker(self, attacker):
        if not isinstance(attacker, _Attacker):
            raise ValueError('attacker must be an attacker class')
        if attacker in self.attackers_noTar:
            raise ValueError('attacker already exist')

    def add_nonTarAttackers(self, attackers):
        if not isinstance(attackers, (list, tuple)):
            attackers = [attackers]
        for attacker in attackers:
            self._isAttacker(attacker)
            self.attackers_noTar.append(attacker)

class _TargetedAttackerMixin(object):

    @property
    def attackers_tar(self):          
        return self._attackers_tar

    def _isAttacker(self, attacker):
        if not isinstance(attacker, _Attacker):
            raise ValueError('attacker must be an attacker class')
        if attacker in self.attackers_tar:
            raise ValueError('attacker already exist')

    def add_tarAttackers(self, attackers):
        if not isinstance(attackers, (list, tuple)):
            attackers = [attackers]
        for attacker in attackers:
            self._isAttacker(attacker)
            self.attackers_tar.append(attacker)

class _DefensorMixin(object):

    @property
    def defensors(self):
        return self._defensors
    
    @property
    def defensor_names(self):
        return [x.__class__.__name__ for x in self.defensors]

class Evaluation(_Evaluation, _NonTargetedAttackerMixin, _TargetedAttackerMixin, _DefensorMixin):

    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)

    @cuda_free_cache
    def _evaluation_attack(self, epsilon, targeted=False, save=False):
        if targeted:
            attackers = self.attackers_tar
        else:
            attackers = self.attackers_noTar
        batches_advExes = []
        evaluation_dfs = [self.evaluation_cleanImages]
        for attacker in attackers:
            batches_adv, evaluation_adv = self._attack(attacker, epsilon, targeted)
            batches_advExes.append(batches_adv)
            evaluation_dfs.append(evaluation_adv)
        evaluation_df = pd.concat(evaluation_dfs, ignore_index=True)
        print(evaluation_df)
        if save:
            if targeted:
                evaluation_df.to_csv('attack_targeted.csv')
            else:
                evaluation_df.to_csv('attack_nonTargeted.csv')

        return batches_advExes

    @cuda_free_cache
    def evaluation(self, save=False):
        epsilon = self._randomly_choosen_epsilon()

        batches_advExes_notargeted = self._evaluation_attack(epsilon, targeted=False, save=save)
        batches_advExes_targeted = self._evaluation_attack(epsilon, targeted=True, save=save)

class _Evaluation_Attack(_Evaluation):

    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)

    def _attacker_visualize(self, attacker, targeted=False):
        inputs, targets = self._batch_parse(self.batches[0], targeted)
        if isinstance(targeted, bool) and not targeted:
            visualize_adversary_batch(self.model, self.device, (inputs, targets), self.class_names, attacker)
        else:
            visualize_targetedAdversary_batch(self.model, self.device, self.batches[0], self.class_names, attacker, targets)

    def _attackers_visualize(self, attackers, targeted=False):
        epsilon = self._randomly_choosen_epsilon()
        for attacker in attackers:
            attacker.epsilon = epsilon
            self._attacker_visualize(attacker, targeted)
  
    def _epsilon_analysis_visualize(self, df_result, names):
        plt.figure(figsize=(20, 10))
        for i, k in enumerate(self.topk):
            plt.plot(self.limited_epsilons, np.ones_like(self.limited_epsilons)*self.acc_cleanImages[i], label=f'Top{k}_clean_images')
        for name in names:
            df = df_result[df_result.Name == name]
            plt.plot(df.Epsilon, df.Accuracy, label=name)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, max(self.limited_epsilons), step=1))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def _epsilon_analysis(self, attackers, targeted=False):
        df_results = [self.evaluation_cleanImages]
        for epsilon in self.limited_epsilons:
            for attacker in attackers:
                _, evaluation_adv = self._attack(attacker, epsilon, targeted)
                df_results.append(evaluation_adv)
        df_result = pd.concat(df_results, ignore_index=True)
        print(df_result)
        self._epsilon_analysis_visualize(df_result, [x.__class__.__name__ for x in attackers])

    def _attack_expression(self, attackers, epsilon=None, targeted=False):
        if not epsilon:
            epsilon = self._randomly_choosen_epsilon()
        df_results = [self.evaluation_cleanImages]
        for attacker in attackers:
            _, evaluation_adv = self._attack(attacker, epsilon, targeted)
            df_results.append(evaluation_adv)
        df_result = pd.concat(df_results, ignore_index=True)
        print(df_result)

    def _attack_blackBox_expression(self, ensemble, attackers, epsilon=None, targeted=False):
        if not epsilon:
            epsilon = self._randomly_choosen_epsilon()
        for attacker in attackers:
            self._blackBox_ensemble(ensemble, attacker, epsilon, targeted)

class Evaluation_Attack_NoTargeted(_Evaluation_Attack, _NonTargetedAttackerMixin):
    
    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)

    @cuda_free_cache
    def visualize_attack(self):
        self._attackers_visualize(self.attackers_noTar, targeted=False)

    @cuda_free_cache
    def epsilon_analysis(self):
        self._epsilon_analysis(self.attackers_noTar, targeted=False)

    @cuda_free_cache
    def attack_expression(self, epsilon=None):
        self._attack_expression(self.attackers_noTar, targeted=False, epsilon=epsilon)

    @cuda_free_cache
    def attack_blackBox_expression(self, ensemble, attackers=[], epsilon=None):
        if not attackers:
            attackers = self.attackers_noTar
        elif set(attackers) < set(range(len(self.attackers_noTar))):
            attackers = [self.attackers_noTar[x] for x in attackers]
        else:
            [self._isAttacker(x) for x in attackers]
        self._attack_blackBox_expression(ensemble, attackers, epsilon=epsilon, targeted=False)

class Evaluation_Attack_Targeted(_Evaluation_Attack, _TargetedAttackerMixin):

    def __init__(self, model=None, device='cpu', batch_size=50):
        super().__init__(model=model, device=device, batch_size=batch_size)

    @cuda_free_cache
    def visualize_attack(self, targeted=True):
        self._attackers_visualize(self.attackers_tar, targeted=targeted)

    @cuda_free_cache
    def epsilon_analysis(self, targeted=True):
        self._epsilon_analysis(self.attackers_tar, targeted=targeted)

    @cuda_free_cache
    def attack_expression(self, targeted='most', epsilon=None):
        self._attack_expression(self.attackers_tar, targeted=targeted, epsilon=epsilon)

    @cuda_free_cache
    def attack_blackBox_expression(self, ensemble, attackers=[], targeted='most', epsilon=None):
        if not attackers:
            attackers = self.attackers_tar
        elif set(attackers) < set(range(len(self.attackers_tar))):
            attackers = [self.attackers_tar[x] for x in attackers]
        else:
            [self._isAttacker(x) for x in attackers]
        self._attack_blackBox_expression(ensemble, attackers, epsilon=epsilon, targeted=targeted)