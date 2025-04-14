import torch


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    #####################################################################################################
    ####################################### Confusion Matrix ############################################
    #####################################################################################################

    def confusion_matrix(self, preds, targets):
        if not len(preds.shape) == len(targets.shape):
            raise ValueError(f"Size of prediction {list(preds.shape)} must match size of targets {list(targets.shape)}")
        
        self.confusion = torch.zeros(targets.shape[0], self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion[:,i,j] = torch.logical_and(targets==i, preds==j).flatten(1).sum(dim=-1)
                
        return self.confusion
        
    #####################################################################################################
    ################################################ IOU  ###############################################
    #####################################################################################################

    def iou_per_class(self):
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        fp = torch.sum(self.confusion, dim=-1) - tp
        fn = torch.sum(self.confusion, dim=-2) - tp

        return torch.nan_to_num(tp / (tp + fp + fn), 1).mean(dim=0)

    def mIOU(self):
        return torch.mean(self.iou_per_class()[1:]).item()


    def fIOU(self):
        pixel_frequency = torch.sum(self.confusion, dim=-1) / torch.sum(self.confusion.flatten(1), dim=-1, keepdim=True)
        return torch.mean(torch.sum(pixel_frequency*self.iou_per_class(), dim=-1)).item()


    #####################################################################################################
    ############################################ Accuracy ###############################################
    #####################################################################################################

    def pixel_accuracy(self):
        print(torch.diagonal(self.confusion, dim1=-2, dim2=-1).sum(dim=-1))
        correct_pixels = torch.diagonal(self.confusion, dim1=-2, dim2=-1).sum(dim=-1)
        total_pixels = torch.sum(self.confusion.flatten(1), dim=-1)
        return torch.mean(correct_pixels / total_pixels, dim=0).item()


    def mean_accuracy(self):
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        pixel_per_class = torch.sum(self.confusion, dim=-1)
        return torch.mean(torch.nan_to_num(tp / pixel_per_class, 1)).item()

    #####################################################################################################
    ####################################### Dice Score ##################################################
    #####################################################################################################

    def dice_per_class(self):
        smooth = 1
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1)
        fp = torch.sum(self.confusion, dim=(-1, -2)) - tp

        return torch.mean((tp + smooth) / (tp + fp + smooth), dim=0)

    def mDice(self):
        return torch.mean(self.dice_per_class()[1:]).item()
    
    def macroDice(self):
        smooth = 1
        tp = torch.diagonal(self.confusion, dim1=-2, dim2=-1).sum(dim=1)
        fp = torch.sum(self.confusion, dim=-1).sum(dim=1) - tp
        fn = torch.sum(self.confusion, dim=-2).sum(dim=1) - tp
        
        return torch.mean((2*tp + smooth) / (2*tp + fp + fn + smooth), dim=0)

    def fDice(self):
        pixel_frequency = torch.sum(self.confusion, dim=-1) / torch.sum(self.confusion.flatten(1), dim=-1, keepdim=True)
        return torch.mean(torch.sum(pixel_frequency*self.dice_per_class(), dim=-1)).item()
        

    #####################################################################################################
    ####################################### All Metrics #################################################
    #####################################################################################################

    def all_metrics(self):
        miou = self.mIOU()
        #fiou = self.fIOU()
        #accuracy = self.pixel_accuracy()
        mean = self.mean_accuracy()
        mdice = self.mDice()
        #fdice = self.fDice()
        
        return miou, mean, mdice
    