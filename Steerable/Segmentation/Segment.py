import torch
from typing import List, Tuple


#####################################################################################################
################################### Create Segmentaion Dataset ######################################
##################################################################################################### 

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_shape,
                 min_num_per_image = 1, max_num_per_image = 1, 
                 max_iou = 0.2, n_samples = None) -> None:

        
        
        self.dataset = dataset
        self.image_shape = tuple(dataset[0][0].shape) if image_shape is None else image_shape
        self.min_num_per_image = min_num_per_image
        self.max_num_per_image = max_num_per_image
        self.max_iou = max_iou
        self.n_samples = len(self.dataset) if n_samples is None else n_samples

    def __getitem__(self, index):
        if 0 <= index < self.n_samples:
            image, target = self.create_semantic_segmentation_data()
        else:
            raise ValueError("Index Out of Bounds.")

        return image, target[0]

    def __len__(self):
        return self.n_samples

    
    def create_semantic_segmentation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:

        num_digits = torch.randint(self.min_num_per_image, self.max_num_per_image + 1, (1,)).item()

        input_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid = self.overlay_arrays(
            num_input_arrays_to_overlay=num_digits)

        target_array = self.create_segmentation_target(images=arrays_overlaid,
                                                    labels=labels_overlaid,
                                                    bounding_boxes=bounding_boxes_overlaid)


        return input_array, target_array


    def create_segmentation_target(self,
                                images: torch.Tensor,
                                labels: torch.Tensor,
                                bounding_boxes: torch.Tensor,
                                ) -> torch.Tensor:

        if len(bounding_boxes) != len(labels) != len(images):
            raise ValueError(
                f'The length of bounding_boxes must be the same as the length of labels. Received shapes: {bounding_boxes.shape}!={labels.shape}')

        # Adding one for background
        individual_targets = [torch.zeros(self.image_shape) for _ in range(len(images)+1)]
        labels.insert(0,-1)
        labels = torch.tensor(labels, dtype=torch.int32) + 1
        for i in range(len(bounding_boxes)):
            xmin, ymin, xmax, ymax = bounding_boxes[i]
            individual_targets[i+1][..., ymin:ymax, xmin:xmax] += images[i]

        targets = torch.stack(individual_targets, dim=0)
        target = labels[torch.argmax(targets, dim=0)]

        return target
    
    def overlay_arrays(self, num_input_arrays_to_overlay: int):

        output_array = torch.zeros(self.image_shape)

        indices = torch.randint(0, len(self.dataset), (num_input_arrays_to_overlay, ))
        bounding_boxes = []
        bounding_boxes_as_tuple = []
        arrays_overlaid = []
        labels_overlaid = []
        
        for i in indices:
            images, labels = self.dataset[i]
            bounding_box = self.overlay_at_random(array1=output_array, array2=images, bounding_boxes=bounding_boxes)
            
            arrays_overlaid.append(images)
            labels_overlaid.append(labels.item())
            
            if bounding_box is None:
                break


            bounding_boxes_as_tuple.append(list(bounding_box.values()))
            bounding_boxes.append(bounding_box)
            
        arrays_overlaid = arrays_overlaid
        labels_overlaid = labels_overlaid
        bounding_boxes_overlaid = bounding_boxes_as_tuple

        return output_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid

    def overlay_at_random(self, array1: torch.Tensor, array2: torch.Tensor,
                      bounding_boxes: List[dict] = None) -> torch.Tensor:
        if not bounding_boxes:
            bounding_boxes = []

        *_ , height1, width1 = array1.shape
        *_ , height2, width2 = array2.shape

        max_x = width1 - width2
        max_y = height1 - height2
        
        is_valid = False
        # This number is arbitrary. There are better ways of doing this but this is fast enough.
        max_attempts = 1000
        attempt = 0
        while not is_valid:
            if attempt > max_attempts:
                return
            else:
                attempt += 1
            x = torch.randint(0, max_x + 1, (1,)).item()
            y = torch.randint(0, max_y + 1, (1,)).item()
            
            candidate_bounding_box = {
                'xmin': x,
                'ymin': y,
                'xmax': x + width2,
                'ymax': y + height2,
            }
    
            is_valid = True
            for bounding_box in bounding_boxes:
                iou = self.calculate_iou(bounding_box, candidate_bounding_box)
                if  iou > self.max_iou:
                    is_valid = False
                    break

        self.overlay_array(array1=array1, array2=array2, x=x, y=y)

        return candidate_bounding_box
    def overlay_array(self, array1: torch.Tensor, array2:torch.Tensor, x: int, y: int) -> torch.Tensor:

        *other1, height1, width1,  = array1.shape
        *other2, height2, width2 = array2.shape

        if height2 > height1 or width2 > width1:
            raise ValueError('array2 must have a smaller shape than array1')

        if other1 != other2:
            raise ValueError('array1 and array2 must have same non-singleton dimension.')

        max_array_value = max([torch.max(array1), torch.max(array2)])
        min_array_value = min([torch.min(array1), torch.min(array2)])
        array1[..., y:y+height2, x:x+width2] += array2

        array1 = torch.clamp_(array1, min_array_value, max_array_value)

        return array1
    
    
    def calculate_iou(self, bounding_box1: dict, bounding_box2: dict) -> float:
        A1 = ((bounding_box1['xmax'] - bounding_box1['xmin'])
            * (bounding_box1['ymax'] - bounding_box1['ymin']))
        A2 = ((bounding_box2['xmax'] - bounding_box2['xmin'])
            * (bounding_box2['ymax'] - bounding_box2['ymin']))

        xmin = max(bounding_box1['xmin'], bounding_box2['xmin'])
        ymin = max(bounding_box1['ymin'], bounding_box2['ymin'])
        xmax = min(bounding_box1['xmax'], bounding_box2['xmax'])
        ymax = min(bounding_box1['ymax'], bounding_box2['ymax'])

        if ymin >= ymax or xmin >= xmax:
            return 0

        return ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)

