# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import time
import copy

from packages.myVisualizeShow import showImageWithLabel

def test_visualize_model(model, device, dataloaders):
    model.eval()
    with torch.no_grad():
        corrects = 0
        show_data = []
        show_labels = []
        show_preds = []
        for i,(data, label) in enumerate(dataloaders['test']):
            inputs = data.to(device)
            labels = label.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            if (i == 0):
                show_data = copy.deepcopy(inputs.detach().cpu())
                show_labels = copy.deepcopy(labels.detach().cpu())
                show_preds = copy.deepcopy(preds.detach().cpu())
        
        acc = corrects.double() / len(dataloaders['test'].dataset)
        class_names = dataloaders['test'].dataset.class_names
        print(f'Test Acc: {acc:.4f}')
        showImageWithLabel(show_data, nrow=4,
                        labels=[class_names[x] for x in show_labels], 
                        preds=[class_names[x] for x in show_preds])

def train_model(model, device, dataloaders, criterion, optimizer, scheduler=None, num_epochs=15):
    print(f'\nBegin Trainning in {model.__class__.__name__}......')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataloaders['sizes'][phase]
            epoch_acc = running_corrects.double() / dataloaders['sizes'][phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    test_visualize_model(model, device, dataloaders)

    return model



# %%
