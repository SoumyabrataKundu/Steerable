# pylint: disable=C,R,E1101
import torch
import os
import numpy as np
from Steerable.datasets.hdf5 import HDF5Dataset

class Cath(torch.utils.data.Dataset):
    def __init__(self, data, mode,
                 discretization_bins,
                 discretization_bin_size=2.0,
                 use_density=True):
        """
        :param dataset: String specifying name of cath set
        :param mode: train, test or val
        :param download: Whether to retrieve dataset automatically
        :param discretization_bins: Number of bins used in each dimension
        :param discretization_bin_size: Size of a bin in each dimension (in Angstrom)
        :param use_density: Whether to populate grid with densities rather than a one-hot encoding
        """
        dirname, dataset = os.path.split(dataset)
        self.root = os.path.expanduser(dirname if dirname != "" else ".")


        self.discretization_bins = discretization_bins
        self.discretization_bin_size = discretization_bin_size
        self.use_density = use_density

        data = np.load(os.path.join(self.root, dataset))
        split_start_indices = data['split_start_indices']
        if mode == 'train':
            split_range = [split_start_indices[0], split_start_indices[7]]
        elif mode == 'val':
            split_range = [split_start_indices[7], split_start_indices[8]]
        elif mode == 'test':
            split_range = [split_start_indices[8], None]
        else:
            raise ValueError('Invalid mode {mode}. Should be one of train, test or val.')
        
        self.positions = data['positions'][split_range[0]:split_range[1]]
        self.atom_types = data['atom_types'][split_range[0]:split_range[1]]
        self.n_atoms = data['n_atoms'][split_range[0]:split_range[1]]
        self.labels = [tuple(v) if len(v) > 1 else v[0] for v in data['labels'][split_range[0]:split_range[1]]]

        self.atom_type_set = np.unique(self.atom_types[0][:self.n_atoms[0]])
        self.n_atom_types = len(self.atom_type_set)
        self.atom_type_map = dict(zip(self.atom_type_set, range(len(self.atom_type_set))))

        self.label_set = sorted(list(set(self.labels)))
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))

    def __getitem__(self, index):

        n_atoms = self.n_atoms[index]
        positions = self.positions[index][:n_atoms]
        atom_types = self.atom_types[index][:n_atoms]
        label = self.label_map[self.labels[index]]

        p = self.discretization_bin_size
        n = self.discretization_bins


        fields = torch.zeros(*(self.n_atom_types,)+(n, n, n))
        a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)

        if self.use_density:
            # Create linearly spaced grid
            a = torch.linspace(start=-n / 2 * p + p / 2, end=n / 2 * p - p / 2, steps=n)

            # Pytorch does not suppoert meshgrid - do the repeats manually
            xx = a.view(-1, 1, 1).repeat(1, len(a), len(a))
            yy = a.view(1, -1, 1).repeat(len(a), 1, len(a))
            zz = a.view(1, 1, -1).repeat(len(a), len(a), 1)

            for i, atom_type in enumerate(self.atom_type_set):

                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]

                # Transfer position vector to gpu
                pos = torch.FloatTensor(pos)
                    
                xx_xx = xx.view(-1, 1).repeat(1, len(pos))
                posx_posx = pos[:, 0].contiguous().view(1, -1).repeat(len(xx.view(-1)), 1)
                yy_yy = yy.view(-1, 1).repeat(1, len(pos))
                posy_posy = pos[:, 1].contiguous().view(1, -1).repeat(len(yy.view(-1)), 1)
                zz_zz = zz.view(-1, 1).repeat(1, len(pos))
                posz_posz = pos[:, 2].contiguous().view(1, -1).repeat(len(zz.view(-1)), 1)

                # Calculate density
                sigma = 0.5*p
                density = torch.exp(-((xx_xx - posx_posx)**2 + (yy_yy - posy_posy)**2 + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))

                # Normalize so each atom density sums to one
                density /= torch.sum(density, dim=0)

                # Sum densities and reshape to original shape
                fields[i] = torch.sum(density, dim=1).view(xx.shape)

        else:

            for i, atom_type in enumerate(self.atom_type_set):
                
                # Extract positions with current atom type
                pos = positions[atom_types == atom_type]
                # Lookup indices and move to GPU
                indices = torch.LongTensor(np.ravel_multi_index(np.digitize(pos, a+p/2).T, dims=(n, n, n)))
                # Set values
                fields[i].view(-1)[indices] = 1

        return fields, label

    def __len__(self):
        return len(self.labels)
    
    
def main(data, bins, use_density):
    filename = 'Cath' + ('_density' if use_density else '') + '_' + str(bins) + '.hdf5'
    hdf5file = HDF5Dataset(filename)
    
    for mode in ['train', 'test', 'val']:
        dataset = Cath(data=data, mode = mode, discretization_bins=bins, discretization_bin_size=2.0, use_density=use_density)
        hdf5file.create_hdf5_dataset(mode, dataset)
    
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--use_density", type=bool, default=True)

    args = parser.parse_args()

    main(**args.__dict__)
