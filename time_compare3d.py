import torch
import torch.nn as nn
import time
from numpy import mean, std

import Steerable.nn as snn


###################################################################################################################
############################################## Model ##############################################################
###################################################################################################################

device = torch.device('cuda')
class Model(torch.nn.Module):
    def __init__(self, n_radius, restricted, conv_first) -> None:
        super().__init__()
        n_theta = 40
        
        self.network = nn.Sequential(
            snn.SE3Conv(1, [2,3], 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first),
            snn.SE3NormNonLinearity([2,3]),
            snn.SE3Conv([2,3], [2,3], 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first),
            snn.SE3BatchNorm(),
            
            snn.SE3Conv([2,3], [1,1], 3, n_radius=n_radius, n_theta=n_theta, padding='same', restricted=restricted, conv_first=conv_first),
            
            snn.SE3NormFlatten()
            
        )
    
    def forward(self,x):
        return self.network(x.type(torch.cfloat))

###################################################################################################################
############################################## Compare Time #######################################################
###################################################################################################################


def time_compare(model):
    forward_pass = []
    backward_pass = []
    total = []
    num_classes = 2
    input_res = [2,1,10,10,10]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    for _ in range(10):
        inputs = torch.randn(*input_res, device=device)
        labels = torch.randint(0,num_classes-1, (input_res[0],), device=device)

        t0 = time.time()
        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        t1 = time.time()

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t2 = time.time()

        forward_pass.append((t1-t0)*1000)
        backward_pass.append((t2-t1)*1000)
        total.append((t2-t0)*1000)
    print(f'{mean(total): .2f} +- {std(total): .2f}')
    
    
def main():
    n_radius = [2, 4]

    for r in n_radius:
        for restricted in [True, False]:
            for conv_first in [True, False]:
                print(f'r{r} restricted {restricted} conv_first {conv_first} : ', end='')
                model = Model(r, restricted=restricted, conv_first=conv_first).to(device)
                time_compare(model)
            print()
        print('\n')

if __name__=='__main__':
    main()