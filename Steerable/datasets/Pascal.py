import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import rotate

from Steerable.datasets.hdf5 import HDF5Dataset
    
    
#####################################################################################################
##################################### Pascal Dataset Class ##########################################
##################################################################################################### 

class Pascal(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', rotate=False, image_transform = None, target_transform = None) -> None:
        self.image_transform = image_transform
        self.target_transform = target_transform
        file_path = os.path.join(data_path, 'ImageSets', mode+'.txt')
        self.rotate = rotate
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            self.image_files = [os.path.join(data_path + '/JPEGImages', line.strip() + '.jpg') for line in lines]
            self.target_files = [os.path.join(data_path + '/SegmentationClass', line.strip() + ".png") for line in lines]
            
        assert len(self.image_files) == len(self.target_files)
        self.n_samples = len(self.image_files)
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        target_file = self.target_files[index]
 
        assert os.path.basename(image_file)[:-4] == os.path.basename(target_file)[:-4]
        
        image = Image.open(image_file)
        target = Image.open(target_file)
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target) * 255
            target[target == 255] = 0
            target = target.type(torch.int)
        
        if self.rotate:
            degree = torch.randint(0, 360, (1,)).item()
            image = rotate(image, degree)
            target = rotate(target, degree)

        return image, target[0]
    
    def __len__(self):
        return self.n_samples
    
    
def get_datasets(data_path, rotate=False) -> dict:
    transformation = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        ])
    
    train_dataset = Pascal(data_path, mode='train', rotate=rotate, image_transform=transformation, target_transform=transformation)
    test_dataset = Pascal(data_path, mode='test', rotate=rotate, image_transform=transformation, target_transform=transformation)
    
    return {'train' : train_dataset, 'test' : test_dataset}
    
    
#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 
    
def main():
    datasets = get_datasets('../data/Pascal', rotate=True)
    hdf5file = HDF5Dataset('Pascal.hdf5')
    
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])
        
    return
    
# if __name__ == "__main__":
#     main()