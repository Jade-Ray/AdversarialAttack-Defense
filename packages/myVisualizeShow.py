import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
import time

from packages.myDecorator import cuda_free_cache

def showImageWithLabel(datas, labels, preds=[], nrow=8):
    data_size = datas[0].size()[1]
    plt.figure(figsize=(8, 10))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('a batch expression')
    plt.imshow(np.transpose(vutils.make_grid(datas, nrow=nrow, padding=2, normalize=True), (1, 2, 0)))
    x_text = 0
    y_text = data_size
    for i in range(datas.size()[0]):
        if (preds == []):
            plt.text(x_text, y_text, labels[i], fontdict={'size': 12, 'color': 'g'})
        else:
            if (preds[i] == labels[i]):
                plt.text(x_text, y_text, labels[i], fontdict={'color': 'g'})
            else:
                plt.text(x_text, y_text, labels[i], fontdict={'color': 'r'})
        x_text += data_size + 2
        if (i + 1) % nrow == 0:
            x_text = 0
            y_text += data_size
    plt.show()

def visualize_examples(examples):
    if len(examples) > 16:
        examples = examples[:16]
    plt.figure(figsize=(10, 15))
    nrow = 4
    ncol = math.ceil(len(examples) / nrow)
    for i, example in enumerate(examples, 1):
        plt.subplot(ncol, nrow, i)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"Prob: {example['prob']:.2f}%")
        plt.xlabel(example['pred'].replace(',', '\n'), fontdict={'color': 'green' if example['correct'] else 'red'})
        plt.imshow(example['image'])
    plt.tight_layout()
    plt.show()

@cuda_free_cache
def visualize_batch(model, device, batch, class_names):
    examples = []    
    model.eval()
    with torch.no_grad():    
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        probs, preds = torch.max(outputs, 1)
        for i in range(len(inputs)):
            example = np.transpose(inputs[i].detach().cpu().numpy(), (1, 2, 0))
            target = targets[i].item()
            pred = preds[i].item()
            correct = target==pred
            prob = probs[i].item() * 100
            print(f'Number {i} input: {correct}\ttarget: {target}\tpred: {pred}\tprob: {prob:.2f}%')
            examples.append({'image': example, 'pred': class_names[pred], 'prob': prob, 'correct': correct})
    visualize_examples(examples)

def visualize_single_batch(model, device, dataloader):
    class_names = dataloader.dataset.class_names
    batch = next(iter(dataloader))
    model.to(device)
    visualize_batch(model, device, batch, class_names)

def visualize_adv_examples(adv_examples):
    if len(adv_examples) > 8:
        adv_examples = adv_examples[:8]
    plt.figure(figsize=(10, 15))
    nrow = 4
    ncol = math.ceil(len(adv_examples) * 2 / nrow)
    cnt = 1
    for example in adv_examples:
        for i in range(2):
            plt.subplot(ncol, nrow, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if i == 0:
                plt.title(f"Prob: {example['init_prob']:.2f}%")
                plt.xlabel(example['init_pred'].replace(',', '\n'))
                plt.imshow(example['init_image'])
            else:
                plt.title(f"Prob: {example['final_prob']:.2f}%")
                plt.xlabel(example['final_pred'].replace(',', '\n'))
                plt.imshow(example['final_image'])
            cnt += 1
    plt.tight_layout()
    plt.show()

@cuda_free_cache
def visualize_targetedAdversary_batch(model, device, batch, class_names, attacker, targeted):
    adv_examples = []  
    print(f'\nAcctack model with {attacker.name} method')
    since = time.time()

    model.eval()
    perturbed_data = attacker(batch[0], targeted)
    with torch.no_grad():
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        init_probs, init_preds = outputs.max(1)
      
        outputs = torch.softmax(model(perturbed_data), dim=1)
        final_probs, final_preds = outputs.max(1)

        init_correct = 0
        final_correct = 0
        for i in range(len(inputs)):
            if init_preds[i] != targets[i]:
                continue
            else:
                init_correct += 1
                if final_preds[i] == targeted[i]:
                    final_correct +=1
                    if len(adv_examples) < 9:
                        init_pred = class_names[init_preds[i].item()]
                        init_prob = init_probs[i].item() * 100
                        final_pred = class_names[final_preds[i].item()]
                        final_prob = final_probs[i].item() * 100
                        example = np.transpose(inputs[i].detach().cpu().numpy(), (1, 2, 0))
                        adv_ex = np.transpose(perturbed_data[i].detach().cpu().numpy(), (1, 2, 0))
                        adv_examples.append({'init_image': example, 'final_image': adv_ex, 'init_pred': init_pred, 'init_prob': init_prob, 'final_pred': final_pred, 'final_prob': final_prob})
        
    totall = len(inputs)
    init_acc = init_correct / totall * 100
    final_acc = final_correct/ totall * 100
    time_elapsed = time.time() - since
    print(f'Spend time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\tClean Images Acc:{init_acc:.2f}%([{init_correct}]/[{totall}])\
        \tTargeted Attack Acc: {final_acc:.2f}%([{final_correct}]/[{totall}])')
    visualize_adv_examples(adv_examples)

@cuda_free_cache
def visualize_adversary_batch(model, device, batch, class_names, attacker):
    adv_examples = []  
    print(f'\nAcctack model with {attacker.name} method')
    since = time.time()

    model.eval()
    perturbed_data = attacker(batch[0], batch[1])
    with torch.no_grad():
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        init_probs, init_preds = outputs.max(1)
      
        outputs = torch.softmax(model(perturbed_data), dim=1)
        final_probs, final_preds = outputs.max(1)

        init_correct = 0
        final_correct = 0
        for i in range(len(inputs)):
            if init_preds[i] != targets[i]:
                continue
            else:
                init_correct += 1
                if final_preds[i] == targets[i]:
                    final_correct +=1
                elif len(adv_examples) < 9:
                    init_pred = class_names[init_preds[i].item()]
                    init_prob = init_probs[i].item() * 100
                    final_pred = class_names[final_preds[i].item()]
                    final_prob = final_probs[i].item() * 100
                    example = np.transpose(inputs[i].detach().cpu().numpy(), (1, 2, 0))
                    adv_ex = np.transpose(perturbed_data[i].detach().cpu().numpy(), (1, 2, 0))
                    adv_examples.append({'init_image': example, 'final_image': adv_ex, 'init_pred': init_pred, 'init_prob': init_prob, 'final_pred': final_pred, 'final_prob': final_prob})
        
    totall = len(inputs)
    init_acc = init_correct / totall * 100
    final_acc = final_correct/ totall * 100
    time_elapsed = time.time() - since
    print(f'Spend time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\tAcc from {init_acc:.2f}%([{init_correct}]/[{totall}]) to {final_acc:.2f}%([{final_correct}]/[{totall}])')
    visualize_adv_examples(adv_examples)

def visualize_adversary_signle_batch(model, device, test_loader, attacker):
    class_names = test_loader.dataset.class_names
    batch = next(iter(test_loader))
    model.to(device)
    visualize_adversary_batch(model, device, batch, class_names, attacker)

@cuda_free_cache
def visualize_classifier_distribution(model, device, batches, class_num=1000):
    classifier_numbers = np.zeros(class_num, dtype=int)
    classifier_accuracies = np.zeros(class_num, dtype=float)
    topk = (1, 5)
    res = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.softmax(model(inputs), dim=1)
            res.append(accuracy(outputs, targets, topk=topk))
            probs, preds = torch.max(outputs, 1)
            for pred in preds:
                classifier_numbers[pred.item()] += 1
            for i in range(outputs.size(0)):
                ori_acc = classifier_accuracies[targets[i].item()]
                classifier_accuracies[targets[i].item()] += outputs[i][targets[i].item()]
                if ori_acc != 0:
                    classifier_accuracies[targets[i].item()] /= 2
    for i, k in enumerate(topk):
        acc = 0
        for r in res:
            acc += r[i].item()
        acc /= len(res)
        print(f'Top-{k} accuracy is {acc:.2f}%')

    plt.figure(figsize=(8, 10))
    plt.subplot(211)
    plt.title(f'Classifier number distribution(with {model.__class__.__name__})')
    plt.bar(range(class_num), classifier_numbers)
    plt.xlabel('Classes')
    plt.ylabel('Pred numbers')

    plt.subplot(212)
    plt.title(f'Classifier accuary distribution(with {model.__class__.__name__})')
    plt.bar(range(class_num), classifier_accuracies)
    plt.xlabel('Classes')
    plt.ylabel('Pred accuracy') 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def batch_corract(model, device, correct, batch, LLT):
    model.eval()
    with torch.no_grad():
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        _, preds = outputs.max(1)
        correct += torch.sum(preds == targets).item()
        if LLT:
            for i, prob in enumerate([outputs[x, targets[x]] for x in range(outputs.size(0))]):
                if prob.item() < LLT[1]:
                    LLT[0], LLT[1] = targets[i].item(), prob.item()
    return correct

def dataloader_acc(model, device, dataloader, attacker=None, LLT=None):
    correct = 0
    totall = len(dataloader.dataset)
    since = time.time()
    for inputs, targets in dataloader:
        if attacker:
            inputs = attacker(inputs, targets)
        correct = batch_corract(model, device, correct, (inputs, targets), LLT)        
    acc = correct / totall
    time_elapsed = time.time() - since
    if attacker:
        print(f'\nAcctack model with {attacker.name} method')
    else:
        print('Clean Image Accuracy')
    if LLT:
        print(f'Least-Likely Class is {LLT[0]}(prob = {LLT[1]*100:.4f}%)')
    print(f'Spend time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\tAcc: {acc*100:.2f}%([{correct}]/[{totall}])')   
    return acc, f'{time_elapsed:.2f}'
