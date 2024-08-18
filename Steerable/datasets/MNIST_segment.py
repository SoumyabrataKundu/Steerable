import torch
import torchvision
import torchvision.transforms as transforms

import Steerable.Segmentation.Segment as Segment
from Steerable.datasets.hdf5 import HDF5Dataset

#####################################################################################################
######################################## MNIST Dataset ##############################################
##################################################################################################### 


class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_path, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test"]:
            raise ValueError("Invalid mode")

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.dataset = torchvision.datasets.MNIST(data_path, train = mode=='train')
        self.n_samples = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset.data[index]
        target = self.dataset.targets[index]
        
        if self.image_transform is not None:
            image = self.image_transform(image.unsqueeze(0)).squeeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image.unsqueeze(0), target

    def __len__(self):
        return self.n_samples


def get_datasets(data_path, rotate=False) -> dict:
    if rotate:
        transformations = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 360), interpolation=transforms.InterpolationMode.NEAREST),
            ])
    else:
        transformations = None
        
    kwargs = {
        'image_shape' : (1,60,60),
        'min_num_digits_per_image' : 2,
        'max_num_digits_per_image' : 4,
        'max_iou' : 0.2,
    }
    
    train_dataset = MNIST(data_path, 'test',  image_transform=transformations)
    test_dataset = MNIST(data_path, 'train', image_transform=transformations)
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [55000, 5000])
    
    train_dataset = Segment(train_dataset, **kwargs)
    val_dataset = Segment(val_dataset, **kwargs)
    test_dataset = Segment(test_dataset, **kwargs)
    
    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}


#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 


def main():
    datasets = get_datasets('../data/MNIST', rotate=True)
    hdf5file = HDF5Dataset('MNIST_segment.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return
    
# if __name__ == "__main__":
#     main()
