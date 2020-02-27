# %% [markdown]
# # Fast Gradient Sign Attack 😉
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import copy

from packages.myGpuDevice import device
from packages.myDatasetsLoader import loader_CIFAR10, loader_MNIST
from packages.myTrainModel import train_model

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "Data/PretrainedModel/lenet_mnist_model.pth"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = LeNet().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.eval()

# %% [markdown]
# ---
# * ***some efficient function***
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def randomly_adding(image, epsilon, data_grad):
    random = torch.rand_like(image)
    perturbed_image = image + epsilon*torch.where(random > 0.5, torch.full_like(random, 1), torch.full_like(random, -1))
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def uniformDis_adding(image, epsilon, data_grad):
    random = (torch.rand_like(image) - 0.5) / 0.5
    perturbed_image = image + epsilon*random
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test(model, test_loader, epsilon, attack_func=fgsm_attack):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = attack_func(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    
    final_acc = correct/float(len(test_loader))
    print(f'Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}')
    
    return final_acc, adv_examples

def getGrad(model, data, target):
    model.eval()
    data, target = data.detach().to(device), target.detach().to(device)
    data.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    return data.grad.data

def adversarial_train(model, train_loader, alpha=0.5, num_epoch=5):
    model.train()
    epsilon = train_loader.dataset.epsilon
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epoch):
        print(f'Epoch {epoch}/{num_epoch - 1}')
        print('-' * 10)
        running_loss = 0.0
        running_data_corrects = 0
        running_perturbation_corrects = 0
        for data, target, perturbation in train_loader:
            data, target, perturbation = data.to(device), target.to(device), perturbation.to(device)
            
            output = model(data.detach())
            _, preds1 = torch.max(output, 1)
            running_data_corrects += torch.sum(preds1 == target.data)
            loss1 = F.nll_loss(output, target)
            
            output = model(perturbation)
            _, preds2 = torch.max(output, 1)
            running_perturbation_corrects += torch.sum(preds2 == target.data)
            loss2 = F.nll_loss(output, target)
                      
            loss = alpha * loss1 + (1 - alpha) * loss2
            model.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_data_acc = running_data_corrects.double() / len(train_loader.dataset)
        epoch_perturbation_acc = running_perturbation_corrects.double() / len(train_loader.dataset)
        print(f'Loss: {epoch_loss:.4f}\tAcc_data: {epoch_data_acc:.4f}\tAcc_perturbation: {epoch_perturbation_acc:.4f}')
    return model

def visualize(accuracies, examples):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, '*-')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

def attack_once(model):
    accuracies = []
    examples = []
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=1, shuffle=True)

    for eps in epsilons:
        acc, ex = test(model, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    visualize(accuracies, examples)

def adversarialExamples_test(model, AE_loader):
    correct = 0
    epsilon = AE_loader.dataset.epsilon
    adv_examples = []

    for data, target, perturbation in AE_loader:
        data, target, perturbation = data.to(device), target.to(device), perturbation.to(device)

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        output = model(perturbation)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbation.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    
    final_acc = correct/float(len(AE_loader))
    print(f'Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(AE_loader)} = {final_acc}')

    plt.figure(figsize=(8, 10))
    for i in range(len(adv_examples)):
        plt.subplot(1, len(adv_examples), i + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.ylabel(f"Eps: {epsilon}", fontsize=14)
        orig, adv, ex = adv_examples[i]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

def calculateAcc(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == target.data)
        acc = correct.double() / len(test_loader.dataset)
        print(f'Acc: {correct}/{len(test_loader.dataset)}=>{acc:.4f}')

def get_rubbishExamples(model, rubbish_loader):
    model.eval()
    rubbish_num = 0
    rub_expamles = []
    with torch.no_grad():
        for data, target in rubbish_loader:
            data, target = data.to(device), target.to(device)
            output = torch.exp(model(data))
            probs, preds = torch.max(output, 1)
            rubbish_num += torch.sum(probs> 0.5)
            if (probs > 0.5) and (len(rub_expamles) < 5):
                rub_ex = data.squeeze().detach().cpu().numpy()
                rub_expamles.append((preds.item(), probs.item(), rub_ex))
    exist_acc = rubbish_num/float(len(rubbish_loader.dataset))
    print(f'rubbish examples existence accuracy: {rubbish_num}/{len(rubbish_loader.dataset)}=>{exist_acc:.4f}')

    plt.figure(figsize=(8, 10))
    plt.title(f"Rubbish Examples")
    for i in range(len(rub_expamles)):
        plt.subplot(1, len(rub_expamles), i + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        pred, prob, ex = rub_expamles[i]
        plt.xlabel(f"pred: {pred}", fontsize=14)
        plt.ylabel(f"prob: {prob:.2f}", fontsize=14)       
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ### 😛Experiment of adversarial examples expressing in MNIST datasets
# ### and comparing the fast gradient sign method and randomly add noise to determine the direction of perturbation is matter most
accuracies = []
examples = []
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])), batch_size=1, shuffle=True)

attack_func = fgsm_attack
print(f'Adversarial examples from: {attack_func.__name__}')
for eps in epsilons:
    acc, ex = test(model, test_loader, eps, attack_func)
    accuracies.append(acc)
    examples.append(ex)

visualize(accuracies, examples)

# %% [markdown]
# ---
#  * 😏***An efficient custome dataset class from MINST datasets, but including the perturbation alread dealt with FGSA***
class AdversarialExampleDataset(torch.utils.data.Dataset):
    def __init__(self, epsilon, datasets, model):
        self.epsilon = epsilon
        self.classes = datasets.classes
        self.data = []
        self.target = []
        self.perturbation = []

        for data, target in datasets:
            self.data.append(data)
            self.target.append(target)
            target = torch.tensor(target)
            gradient = getGrad(model, data.unsqueeze(0), target.unsqueeze(0)).squeeze(0).cpu()
            self.perturbation.append(fgsm_attack(data, self.epsilon, gradient))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target, perturbation = self.data[index], self.target[index], self.perturbation[index]
        return data, target, perturbation

MNIST_datasets = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
AE_loader = torch.utils.data.DataLoader(
    AdversarialExampleDataset(epsilon=0.25, datasets=MNIST_datasets, model=model), 
    batch_size=1, shuffle=True)

# %% [markdown]
# ---
# ### 🧐🧐🧐Experiment of training model with adversarial examples
# ### and the cost = αJ(θ, x, y) + (1-α)J(θ, x+εη, y), default α=0.5
# ### viewing whether the trained model can resist adversarial examples
train_MNIST_datasets = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
train_loader = torch.utils.data.DataLoader(
    AdversarialExampleDataset(epsilon=0.25, datasets=train_MNIST_datasets, model=model),
    batch_size=64, shuffle=True)

model2 = copy.deepcopy(model)
model2 = adversarial_train(model2, train_loader, num_epoch=25)
attack_once(model2)

# %% [markdown]
# ---
# ### 🥳🥳🥳Experiment of the expression of different models in same adversarial examples
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

models = []
for i in range(2):
    m = CustomNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    m = train_model(m, device, loader_MNIST, criterion, optimizer, num_epochs=15)
    models.append({'name': f'custom model {i}', 'model': m})

for i in range(2):
    m = LeNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
    m = train_model(m, device, loader_MNIST, criterion, optimizer, num_epochs=15)
    models.append({'name': f'model {i}', 'model': m})

models.append({'name': f'origin model', 'model': model})

for m in models:
    print(f'adversarial examples in {m["name"]}')
    print('-'*15)
    adversarialExamples_test(m['model'], AE_loader)
    print()

# %%[markdown]
# ---
# * 👌***An efficient custome dataset class from MINST datasets, but replacing data with gradient***
class GradientDataset(torch.utils.data.Dataset):
    def __init__(self, epsilon, datasets, model):
        self.epsilon = epsilon
        self.classes = datasets.classes
        self.data = []
        self.target = []

        for data, target in datasets:
            self.target.append(target)
            target = torch.tensor(target)
            gradient = getGrad(model, data.unsqueeze(0), target.unsqueeze(0)).squeeze(0).cpu()
            self.data.append(self.epsilon * gradient.sign())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        return data, target

gradient_loader = torch.utils.data.DataLoader(
    GradientDataset(epsilon=1, datasets=MNIST_datasets, model=model), 
    batch_size=1, shuffle=True)

# %% [markdown]
# ---
# ### 🤠🤠🤠Experiment of rubbish class examples
# ### and similarity whith adversarial exammples
rubbish_loader = torch.utils.data.DataLoader(
    datasets.FakeData(size=10000, image_size=(1, 28, 28), num_classes=10, transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=1, shuffle=True)

print('Test in randomly generated images datasets')
get_rubbishExamples(model, rubbish_loader)
print('Test in gradient of MNIST dataset datasets')
get_rubbishExamples(model, gradient_loader)

# %%
