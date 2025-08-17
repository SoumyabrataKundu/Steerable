import torch


class Augment(torch.utils.data.Dataset):
    def __init__(self, dataset, function, parameters):
        self.dataset = dataset
        self.function = function
        self.parameters = parameters
        
    def __getitem__(self, index):
        i = index // len(self.parameters)
        j = index % len(self.parameters)
        
        return self.function(self.dataset[i], self.parameters[j])
        
    def __len__(self):
        return len(self.dataset) * len(self.parameters)