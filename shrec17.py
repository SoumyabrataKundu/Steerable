import torch
import numpy as np
import pandas as pd
import os
import subprocess
import random
import sys

from Steerable.datasets import HDF5Dataset



# Dataset Generation
class SHREC17(torch.utils.data.Dataset):
    def __init__(self, data_path, size, mode='train', image_transform=None, target_transform=None, perturbed=True) -> None:
        
        data = pd.read_csv(os.path.join(data_path, 'all.csv'), delimiter = ",", dtype = str)
        if mode in ['train', 'test', 'val']:
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode {mode}. Should be one of train, test or val.')
        self.class_names = list(data['synsetId'].unique())
        self.size = size
        self.data_path = data_path
        self.data = data[data['split'] == self.mode]
        self.n_samples = len(self.data)
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.perturbed = perturbed

    def __getitem__(self, index):
        id, synsetId = self.data.iloc[index][['id', 'synsetId']]
        location = os.path.join(self.data_path, self.mode + ('_perturbed' if self.perturbed else ''), id + '.obj')
        image = torch.tensor(self.objtovoxel(location))
        target = torch.tensor(self.class_names.index(synsetId))
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
        
        
    def objtovoxel(self, location):
        tmpfile = '%030x.npy' % random.randrange(16**30)
        command = ["obj2voxel", "--size", str(self.size), location, tmpfile]
        subprocess.run(command)
        image = np.load(tmpfile).astype(np.int8).reshape((1, *[self.size]*3))
        os.remove(tmpfile)
        
        return image


    def __len__(self):
        return self.n_samples
    
def get_datasets(data_path, size):
    train_dataset = SHREC17(data_path=data_path, mode = 'train', size=size)
    val_dataset = SHREC17(data_path=data_path, mode = 'val', size=size)
    test_dataset = SHREC17(data_path=data_path, mode = 'test', size=size)
    
    return {'train':train_dataset, 'val':val_dataset, 'test':test_dataset}

def main(data_path, size):
    hdf5file = HDF5Dataset('SHREC17_perturbed64', get_datasets(data_path, size))
    hdf5file.create_hdf5_dataset()


if __name__== '__main__':
    
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)

    args = parser.parse_args()

    main(**args.__dict__)