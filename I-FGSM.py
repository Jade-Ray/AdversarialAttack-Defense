# %% [markdown]
# # Iterative FGSM Attack 😜
import torch
import numpy as np
import torchvision

from packages.myGpuDevice import device
from packages.attacker import Attacker_FGSA_physical, Attacker_IFGSM
from packages.evaluation import Evaluation_Attack_NoTarget

# %% [markdown]
# ### 💕💕💕Experiment of the expression with different epsilon
model = torchvision.models.inception_v3()
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth', 
                                                model_dir='./Data/PretrainedModel')
model.load_state_dict(state_dict)

# %%
evaluation = Evaluation_Attack_NoTarget(model, device, batch_size=15)

# %%
attacker1 = Attacker_FGSA_physical(model, device)

# %%
attacker2 = Attacker_IFGSM(model, device)

# %%
evaluation.add_attacker([attacker1, attacker2])

# %%
evaluation.attackers_visualize()

# %%
evaluation.noTarget_evaluation()

# %%
