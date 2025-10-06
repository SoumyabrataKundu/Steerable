import numpy as np
import pickle
import os
from math import sqrt

def main():
    # Get all files in the directory
    all_files = [os.path.join('./log1', f) for f in os.listdir('./log1')]
    channels, kernels, n_radius, max_m = [], [], [], [] 
    for f in all_files:
        channel, kernel, r, k = f.split('_')[1:5]
        channels.append(int(channel))
        kernels.append(int(kernel))
        n_radius.append(int(r))
        max_m.append(int(k))

    
    channels = list(np.unique(channels))
    kernels = list(np.unique(kernels))
    n_radius = list(np.unique(n_radius))
    max_m = list(np.unique(max_m))
    print(channels, n_radius, max_m)
    times = np.zeros((len(channels), len(kernels), len(n_radius), len(max_m), 2, 2))
    for i,channel in enumerate(channels):
        for j,kernel in enumerate(kernels):
            for x,r in enumerate(n_radius):
                for y,k in enumerate(max_m):
                    for restricted in [False]:
                        for conv_first in [False, True]:
                            with open(f"./log1/times_{channel}_{kernel}_{r}_{k}_{restricted}_{conv_first}.pkl", 'rb') as file:
                                t = pickle.load(file)
                                times[i, j, x, y, int(conv_first), 0] = np.mean(t)
                                times[i, j, x, y, int(conv_first), 1] = 1.28 * np.std(t) / sqrt(len(t))
                                
    with open("times3d.pkl", 'wb') as file:  
        pickle.dump(times, file) 
        
if __name__ == '__main__':
    main()
