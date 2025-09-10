import torch

from Steerable.nn.utils import rotate_image

class RandomRotate(torch.utils.data.Dataset):
    def __init__(self, dataset:torch.utils.data.Dataset, order=1, batched=False):
        self.dataset = dataset
        self.order = order
        self.batched = batched
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        if inputs.ndim == (3+int(self.batched)):
            degree = torch.randint(0, 360, (1,)).item()
            inputs = rotate_image(inputs, degree, order=self.order, batched=self.batched)
            if targets.ndim == (2+int(self.batched)):
                targets = rotate_image(targets, degree, order=0, batched=self.batched)
        elif inputs.ndim == (4+int(self.batched)):
            degree = torch.randint(0, 360, (3,))
            inputs = rotate_image(inputs, degree, order=self.order, batched=self.batched)
            if targets.ndim == (3+int(self.batched)):
                targets = rotate_image(targets, degree, order=0, batched=self.batched)
        else:
            ValueError("Only 2D or 3D image data are supported.")
        return inputs, targets
 
    def __len__(self):
        return len(self.dataset)
