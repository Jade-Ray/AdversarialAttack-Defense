import torch
import torch.nn as nn
import math

class _ModelMixin(object):

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        if value.type not in ['cuda', 'cpu']:
            raise ValueError('device must be cuda or cpu')
        self._device = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not value:
            raise ValueError('model can not be Null')
        value.to(self.device)
        value.eval()
        self._model = value

class _Attacker(_ModelMixin):

    def __init__(self, model, device='cpu'):       
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.model = model

    @property
    def criterion(self):
        return self._criterion
    
    @criterion.setter
    def criterion(self, value):
        self._criterion = value

    def __call__(self, inputs, targets):
        return self.attack(inputs, targets)

    def attack(self, inputs, targets):
        pass

class _PhysicalMixin(object):
    
    @property
    def bit_width(self):
        return self._bit_width

    @bit_width.setter
    def bit_width(self, value):
        if value not in [255]:
            raise ValueError('bit_width must be 255')
        self._bit_width = value

class Attacker_FGSA(_Attacker):

    def __init__(self, model, device='cpu', epsilon=0.005):
        super(Attacker_FGSA, self).__init__(model, device=device)
        self.epsilon = epsilon
    
    @property
    def name(self):
        return f'Fast Gradient Sign Attack(epsilon value is {self.epsilon})'

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    def attack(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        return self.fgsm_attack(inputs, self.epsilon, data_grad)

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

class Attacker_FGSA_physical(Attacker_FGSA, _PhysicalMixin):

    def __init__(self, model, device='cpu', epsilon=2, bit_width=255):
        self.bit_width = bit_width
        super().__init__(model, device=device, epsilon=epsilon)       
    
    @property
    def name(self):
        return f'Fast Gradient Sign Attack in phsical(per-pixel only change 1/{self.bit_width}, epsilon={self.epsilon:.4f})'

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not isinstance(value, int):
            raise ValueError('epsilon must be an integer')
        if value < 2 or value > 128:
            raise ValueError('epsilon must between 2~128')
        self._epsilon = value / self.bit_width

class Attacker_IFGSM(_Attacker, _PhysicalMixin):

    def __init__(self, model, device='cpu', alpha=1, epsilon=2, bit_width=255):
        super(Attacker_IFGSM, self).__init__(model, device=device)
        self.alpha = alpha
        self.bit_width = bit_width
        self.epsilon = epsilon       

    @property
    def name(self):
        return f'Iterative FGSM(alpha={self.alpha}, epsilon={self.epsilon:.4f}, bit_width={self.bit_width})'
   
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if value < 1:
            raise ValueError('alpha must be greater than 1')
        self._alpha = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not isinstance(value, int):
            raise ValueError('epsilon must be an integer')
        if value < 2 or value > 128:
            raise ValueError('epsilon must between 2~128')
        self._epsilon = value / self.bit_width

    @property
    def num_iter(self):
        return min(self.epsilon * self.bit_width + 4, round(1.25 * self.epsilon * self.bit_width))

    def attack(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        return self._attack_iter(0, inputs, targets, inputs)

    def _attack_iter(self, num, inputs, targets, perturbed_image):
        if num == self.num_iter:
            return perturbed_image
       
        perturbed_image.requires_grad = True
        outputs = self.model(perturbed_image)
        loss = self.criterion(outputs, targets)
        self.model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data.sign()
        perturbed_image = torch.add(perturbed_image, self.alpha, data_grad)
        perturbed_image = self._clip(inputs, perturbed_image)

        return self._attack_iter(num + 1, inputs, targets, perturbed_image.detach())

    def _clip(self, inputs, perturbed_image):
        perturbed_image = torch.where(perturbed_image > inputs - self.epsilon, perturbed_image, inputs - self.epsilon)
        perturbed_image = torch.where(perturbed_image < inputs + self.epsilon, perturbed_image, inputs + self.epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image