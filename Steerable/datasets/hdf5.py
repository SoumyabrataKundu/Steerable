import torch
import os
import h5py


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
        self._initialize_hdf5_dataset(name, input.shape, target.shape)
        
        for index, (input, target) in enumerate(dataset):
            try:
                self._write_into_hdf5_file(name, input, target)
            except Exception as e:
                print(f'Excpetion at {index + 1} : {e}')
                
            print(f"{index+1} / {len(dataset)}", end="\r")
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



                
    
