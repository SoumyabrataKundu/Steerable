import torch
import os
import h5py

#####################################################################################################
################################### Create HdF5 dataset #############################################
#####################################################################################################


class HDF5Dataset:
    def __init__(self, filename : str, datasets : dict, overwrite=False) -> None:
        self.filename = filename
        self.datasets = datasets
        
        if not os.path.isfile(self.filename) or overwrite:
            self.create_hdf5_file()
        else:
            raise LookupError("File Already Exists! Do overwrite it set overwrite=True")
        
        

    def create_hdf5_file(self):
        
        
        with h5py.File(self.filename, 'w') as f:
            for mode in self.datasets:
                print(f'Creating {mode} dataset: ', end='')
                try:
                    image, target = self.datasets[mode][0]   
                    f.create_dataset(mode + '_inputs', (0, ) + image.shape, maxshape=(None,) +  image.shape, chunks=True)
                    f.create_dataset(mode + '_targets', (0,) + target.shape, maxshape=(None,) +  target.shape, chunks=True)
                    print('Success!')
                except Exception as e:
                    print(e)
                
        return
                

    def create_hdf5_dataset(self):
        self.file = h5py.File(self.filename, 'a')
        for mode in self.datasets:
            print(f"Mode : {mode} ...")
            dataset = self.datasets[mode]
            
            if self.datasets[mode] is not None:
                for index in range(len(dataset)):
                    try:
                        image, target = dataset[index]
                        self.write_into_hdf5_file(mode, image, target)
                    except Exception as e:
                        print(f'Excpetion as {index + 1} : {e}')
                    print(f"{index+1} / {len(dataset)}", end="\r")
            print('Done')
        self.file.close()
        
        return


    def write_into_hdf5_file(self, mode : str, input : torch.Tensor, target : torch.Tensor):
        inputs = self.file[mode + '_inputs']
        targets = self.file[mode + '_targets']

        inputs.resize((len(inputs) + 1,) + tuple(input.size()))
        targets.resize((len(targets) + 1,) + tuple(target.size()))
        
        
        inputs[-1] = input
        targets[-1] = target
        
        return 



                
    