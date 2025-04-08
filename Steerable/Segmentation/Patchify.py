import torch


class PatchifyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_size):
        self.patch_size = patch_size
        self.dataset = dataset
        self.patches_per_image = len(self.extract_patches(self.dataset[0][0]))
        
    def __len__(self):
        return self.patches_per_image*len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index // self.patches_per_image]
        extracted_patches = self.extract_patches(image)[index % self.patches_per_image]
        extracted_targets = self.extract_patches(target.unsqueeze(0))[index % self.patches_per_image,0]
        return extracted_patches, extracted_targets
    
    def extract_patches(self, tensor):
        patches = tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).flatten(0,1)
        
        return patches
