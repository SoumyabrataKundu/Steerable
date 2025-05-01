import torch

class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        
    #####################################################################################################
    ####################################### Confusion Matrix ############################################
    #####################################################################################################

    def get_confusion_matrix(self, preds, targets):
        if preds is None and targets is None:
            return self.confusion
        elif preds is not None and targets is not None:
            if not preds.shape == targets.shape:
                raise ValueError(f"Size of prediction {list(preds.shape)} must match size of targets {list(targets.shape)}")
            
            conf = torch.zeros(targets.shape[0], self.num_classes, self.num_classes, dtype=torch.int64)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    conf[:,i,j] = torch.logical_and(targets==i, preds==j).flatten(1).sum(dim=-1)
            return conf
        else:
            raise ValueError('Either both preds and targets should be given or none of them.')
        
    def add_to_confusion_matrix(self, preds, targets):
        self.confusion += self.get_confusion_matrix(preds, targets).sum(dim=0)

    #####################################################################################################
    ####################################### Dice Score ##################################################
    #####################################################################################################
    
    def dice_per_class(self, preds=None, targets=None):
        conf = self.get_confusion_matrix(preds, targets)
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)
        fp = torch.sum(conf, dim=-1) - tp
        fn = torch.sum(conf, dim=-2) - tp

        dice = torch.nan_to_num(2*tp / (2*tp + fp + fn), 1)
        return dice if preds is None else dice.mean(dim=0)

    
    def mDice(self, preds=None, targets=None):
        return torch.mean(self.dice_per_class(preds, targets)[1:]).item()
    
    def MDice(self):
        conf = self.confusion
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)[1:].sum()
        fp = (torch.sum(conf, dim=-1) - tp)[1:].sum()
        fn = (torch.sum(conf, dim=-2) - tp)[1:].sum()
        
        return torch.nan_to_num(2*tp / (2*tp + fp + fn), 1).item()
        
    #####################################################################################################
    ################################################ IOU ################################################
    #####################################################################################################
    def iou_per_class(self, preds=None, targets=None):
        conf = self.get_confusion_matrix(preds, targets)
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)
        fp = torch.sum(conf, dim=-1) - tp
        fn = torch.sum(conf, dim=-2) - tp

        iou = torch.nan_to_num(tp / (tp + fp + fn), 1)
        return iou if preds is None else iou.mean(dim=0)
    
    def mIOU(self, preds=None, targets=None):
        return torch.mean(self.iou_per_class(preds, targets)[1:]).item()
    
    def MIOU(self):
        conf = self.confusion
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)[1:].sum()
        fp = (torch.sum(conf, dim=-1) - tp)[1:].sum()
        fn = (torch.sum(conf, dim=-2) - tp)[1:].sum()
        
        return torch.nan_to_num(tp / (tp + fp + fn), 1).item()