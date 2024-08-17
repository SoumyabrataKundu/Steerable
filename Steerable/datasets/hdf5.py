import torch
import os
import h5py

#####################################################################################################
################################### Create HdF5 dataset #############################################
#####################################################################################################


class HDF5Dataset:
    def __init__(self, filename : str, overwrite=False) -> None:
        self.filename = filename
        
        if not os.path.isfile(self.filename) or overwrite:
            self.create_hdf5_file()
        else:
            raise LookupError("File Already Exists! Do overwrite it set overwrite=True")
        
        

    def create_hdf5_file(self, datasets):
        image, target = datasets[list(datasets.keys())[0]][0]
        image_shape, target_shape = image.shape, target.shape
        
        with h5py.File(self.filename, 'w') as f:
            for mode in datasets:    
                f.create_dataset(mode + '_inputs', (0, ) + image_shape, maxshape=(None,) +  image_shape, chunks=True)
                f.create_dataset(mode + '_targets', (0,) + target_shape, maxshape=(None,) +  target_shape, chunks=True)
                
        return
                

    def create_hdf5_dataset(self, datasets):
        self.file = h5py.File(self.filename, 'a')
        for mode in datasets:
            print(f"Mode : {mode} ...")
            dataset = datasets[mode]
            
            if datasets[mode] is not None:
                for index in range(len(dataset)):
                    try:
                        image, target = dataset[index]
                        self.write_into_hdf5_file(mode, image, target)
                    except Exception as e:
                        print(e)
                    print(f"{index+1} / {len(dataset)}", end="\r")
            print('Done')
        self.file.close()
        
        return


    def write_into_hdf5_file(self, mode : str, image : torch.Tensor, target : torch.Tensor):
        images = self.file[mode + '_images']
        targets = self.file[mode + '_targets']

        images.resize((len(images) + 1,) + tuple(image.size()))
        targets.resize((len(targets) + 1,) + tuple(target.size()))
        
        images[-1] = image
        targets[-1] = target
        
        return 



                
    