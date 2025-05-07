import torch
import time
import sys
sys.path.append('../')
import Steerable.nn as snn
import pickle


class Model(torch.nn.Module):
    def __init__(self, channel, kernel, n_radius, max_m, restricted=False, conv_first=False) -> None:
        super().__init__()
        n_theta = 40
        self.network = torch.nn.Sequential(
            snn.SE2ConvType1(1, channel, kernel, n_radius=n_radius, n_theta=n_theta, max_m=max_m, conv_first = conv_first),
            snn.SE2ConvType2(channel, channel, kernel, n_radius=n_radius, n_theta=n_theta, max_m=max_m, restricted=restricted, conv_first = conv_first),
            snn.SE2ConvType2(channel, channel, kernel, n_radius=n_radius, n_theta=n_theta, max_m=max_m, restricted=restricted, conv_first = conv_first),            
        )
    
    def forward(self,x):
        return self.network(x.type(torch.cfloat))

device = 'cuda'
    
def time_compare(channel, kernel, n_radius, max_m, restricted, conv_first):
    resolution = 28
    batch_size = 10
    n_simulation = 100
    burnout = 5
    
    times = []
    restricted = bool(restricted)
    conv_first = bool(conv_first)
    
    model = Model(channel, kernel, n_radius, max_m, restricted, conv_first).to(device)
    inputs = torch.randn(batch_size, 1, *[resolution]*2, device=device)
    
    for sim in range(-burnout, n_simulation):
        t0 = time.time()
        model(inputs)
        t1 = time.time()
        if sim >=0:
            times.append((t1-t0)*1000 / batch_size)

    with open(f"./log/times_{channel}_{kernel}_{n_radius}_{max_m}_{restricted}_{conv_first}.pkl", 'wb') as file:  
        pickle.dump(times, file) 
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--channel", type=int,required=True)
    parser.add_argument("--kernel", type=int,required=True)
    parser.add_argument("--n_radius", type=int,required=True)
    parser.add_argument("--max_m", type=int, required=True)
    parser.add_argument("--restricted", type=int, required=True)
    parser.add_argument("--conv_first", type=int, required=True)

    args = parser.parse_args()
    time_compare(**args.__dict__)
    
