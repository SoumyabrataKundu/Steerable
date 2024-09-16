import torch


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = torch.zeros(self.num_classes, self.num_classes)
    
    #####################################################################################################
    ####################################### Confusion Matrix ############################################
    #####################################################################################################

    def add_to_confusion_matrix(self, preds, targets):
        if not len(preds.shape) == len(targets.shape):
            raise ValueError(f"Size of prediction {list(preds.shape)} must match size of targets {list(targets.shape)}")
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion[i,j] += torch.sum(torch.logical_and(targets==i, preds==j)).item()
        
        
    #####################################################################################################
    ################################################ IOU  ###############################################
    #####################################################################################################

    def iou_per_class(self):
        smooth = 1
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        fp = torch.sum(self.confusion, dim=-1) - tp
        fn = torch.sum(self.confusion, dim=-2) - tp
        return (tp + smooth) / (tp + fp + fn + smooth)


    def mIOU(self):
        return torch.mean(self.iou_per_class()).item()


    def fIOU(self):
        pixel_frequency = torch.sum(self.confusion, dim=-1) / torch.sum(self.confusion).item()
        return torch.sum(pixel_frequency*self.iou_per_class()).item()


    #####################################################################################################
    ############################################ Accuracy ###############################################
    #####################################################################################################

    def pixel_accuracy(self):
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        total_pixels = torch.sum(self.confusion).item()
        return torch.sum(tp).item() / total_pixels


    def mean_accuracy(self):
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        pixel_per_class = torch.sum(self.confusion, dim=-1)
        return torch.mean(tp / pixel_per_class).item()

    #####################################################################################################
    ####################################### Dice Score ##################################################
    #####################################################################################################

    def dice_per_class(self):
        smooth = 1
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        fp = torch.sum(self.confusion, dim=-1) - tp
        fn = torch.sum(self.confusion, dim=-2) - tp

        return (2*tp + smooth) / (2*tp + fp + fn + smooth)

    def mDice(self):
        return torch.mean(self.dice_per_class()).item()

    def fDice(self):
        pixel_frequency = torch.sum(self.confusion, dim=-1) / torch.sum(self.confusion).item()
        return torch.sum(pixel_frequency*self.dice_per_class()).item()
        

    #####################################################################################################
    ####################################### All Metrics #################################################
    #####################################################################################################

    def all_metrics(self):
        miou = self.mIOU()
        fiou = self.fIOU()
        accuracy = self.pixel_accuracy()
        mean = self.mean_accuracy()
        mdice = self.mDice()
        fdice = self.fDice()
        
        return miou, fiou, accuracy, mean, mdice, fdice
    