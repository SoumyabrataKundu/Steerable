import os
import h5py
import torch

class HDF5Dataset:
    def __init__(self, filename : str, overwrite=False) -> None:
        self.filename = filename
        self.overwrite = overwrite
        
        self._create_hdf5_file()
        

    def _create_hdf5_file(self):
        if not os.path.isfile(self.filename) or self.overwrite:
            f = h5py.File(self.filename, 'w')
            f.close()
            
        else:
            raise LookupError("File Already Exists! To overwrite it, set overwrite=True")
                
        return
    
    def _initialize_hdf5_dataset(self, name, input_shape, target_shape):
        try:
            print(f'Creating {name} dataset: ', end='')
            self.file.create_dataset(name + '_inputs', (0, ) + input_shape, maxshape=(None,) +  input_shape, chunks=True)
            self.file.create_dataset(name + '_targets', (0,) + target_shape, maxshape=(None,) +  target_shape, chunks=True)
            print('Success!')
        except Exception as e:
            print(e)
        return

    def create_hdf5_dataset(self, name, dataset):
        self.file = h5py.File(self.filename, 'a')
        input, target = dataset[0]
        target = torch.tensor(target)
        self._initialize_hdf5_dataset(name, input.shape, target.shape)
        
        for index in range(len(dataset)):
            try:
                input, target = dataset[index]
                self._write_into_hdf5_file(name, input, torch.tensor(target))
            except Exception as e:
                print(f'Excpetion at {index + 1} : {e}')
                
            print(f"Writing into {name} dataset : {index+1} / {len(dataset)}", end="\r")
            
        print('Done')
        self.file.close()
        
        return

    def _write_into_hdf5_file(self, name : str, input, target):
        inputs = self.file[name + '_inputs']
        targets = self.file[name + '_targets']

        inputs.resize((len(inputs) + 1,) + tuple(input.shape))
        targets.resize((len(targets) + 1,) + tuple(target.shape))
        
        
        inputs[-1] = input
        targets[-1] = target
        
        return 
    
    
class HDF5(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test", "val"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # Reading from file
        input = torch.from_numpy(self.file[self.mode + '_inputs'][index]).float()
        target = torch.tensor(self.file[self.mode + '_targets'][index]).long()

        # Applying trasnformations
        if self.image_transform is not None:
            input = self.image_transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.file[self.mode+'_targets'])
