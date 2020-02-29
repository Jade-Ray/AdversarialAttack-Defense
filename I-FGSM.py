# %% [markdown]
# # Iterative FGSM Attack 😜
import torch
import numpy as np
import torchvision

from packages.myGpuDevice import device
from packages.attacker import FGSA_NonTargeted, FGSA_Targeted, IFGSM_NonTargeted, IFGSM_Targeted
from packages.evaluation import Evaluation_Attack_NoTargeted, Evaluation_Attack_Targeted

model = torchvision.models.inception_v3()
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth', 
                                                model_dir='./Data/PretrainedModel')
model.load_state_dict(state_dict)

model_resnet50 = torchvision.models.resnet50()
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', 
                                                model_dir='./Data/PretrainedModel')
model_resnet50.load_state_dict(state_dict)

model_vgg19 = torchvision.models.vgg19()
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 
                                                model_dir='./Data/PretrainedModel')
model_vgg19.load_state_dict(state_dict)

evaluation_noTar = Evaluation_Attack_NoTargeted(model, device, batch_size=20)
attacker1 = FGSA_NonTargeted(model, device)
attacker2 = IFGSM_NonTargeted(model, device)
evaluation_noTar.add_nonTarAttackers([attacker1, attacker2])

evaluation_targeted = Evaluation_Attack_Targeted(model, device, batch_size=20)
attacker3 = IFGSM_Targeted(model, device)
attacker4 = FGSA_Targeted(model, device)
evaluation_targeted.add_tarAttackers([attacker3, attacker4])

# %% [markdown]
# ### 💕💕💕Experiment of the expression with different epsilon
evaluation_noTar.visualize_attack()

evaluation_noTar.epsilon_analysis()

# %% [markdown]
# ### 💕💕💕Experiment of targetedAttack
evaluation_targeted.visualize_attack(targeted='most')

evaluation_targeted.epsilon_analysis()

# %% [markdown]
# ### 💕💕💕Experiment of blackBox attack
evaluation_noTar.attack_blackBox_expression([model_resnet50, model_vgg19], epsilon=4)
evaluation_targeted.attack_blackBox_expression([model_resnet50, model_vgg19], epsilon=4, targeted='most')