import torch



#####################################################################################################
####################################### Confusion Matrix ############################################
#####################################################################################################

def confusion_matrix(preds, targets, num_classes):
    if not len(preds.shape) == len(targets.shape):
        raise ValueError(f"Size of prediction {list(preds.shape)} must match size of targets {list(targets.shape)}")
    
    confusion = torch.zeros(*targets.shape[:-2], num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion[...,i,j] = torch.sum(torch.logical_and(targets==i, preds==j), dim=(-2,-1))
            
    return confusion

#####################################################################################################
################################################ IOU  ###############################################
#####################################################################################################

def IOU_per_class(preds, targets, num_classes):
    confusion = confusion_matrix(preds, targets, num_classes)
    tp = torch.diagonal(confusion, dim1=-2, dim2=-1)
    fp = torch.sum(confusion, dim=-1) - tp
    fn = torch.sum(confusion, dim=-2) - tp

    iou = tp / (tp + fp + fn)
    
    return iou

def mIOU_per_class(preds, targets, num_classes):
    iou = IOU_per_class(preds, targets, num_classes)
    
    iou = torch.nanmean(iou.reshape(-1, iou.shape[-1]), dim=0)
    
    return iou


def mIOU(preds, targets, num_classes):
    iou = IOU_per_class(preds, targets, num_classes)
    miou = torch.nanmean(iou.flatten()).item()
    
    return miou



def fIOU(preds, targets, num_classes):
    confusion = confusion_matrix(preds, targets, num_classes)
    iou = IOU_per_class(preds, targets, num_classes)

    
    pixel_per_class = torch.sum(confusion, dim=-2)
    total_pixels = torch.sum(pixel_per_class, dim=(-1,), keepdim=True)
    pixel_frequency = pixel_per_class / total_pixels
    
    
    
    fiou = torch.mean(torch.nansum(pixel_frequency*iou, dim=-1).flatten()).item()
    
    return fiou


#####################################################################################################
############################################ Accuracy ###############################################
#####################################################################################################

def pixel_accuracy(preds, targets, num_classes):
    confusion = confusion_matrix(preds, targets, num_classes)
    tp = torch.diagonal(confusion, dim1=-2, dim2=-1)
    total_pixels = torch.sum(confusion, dim=(-1,-2))
    accuracy = torch.mean(torch.sum(tp, dim=-1) / total_pixels).item()
    
    return accuracy


def mean_accuracy(preds, targets, num_classes):
    confusion = confusion_matrix(preds, targets, num_classes)
    tp = torch.diagonal(confusion, dim1=-2, dim2=-1)
    pixel_per_class = torch.sum(confusion, dim=-2)
    
    mean_accuracy = torch.mean(torch.nanmean(tp / pixel_per_class, dim=-1)).item()
    
    return mean_accuracy

#####################################################################################################
####################################### Dice Score ##################################################
#####################################################################################################

def dice_per_class(preds, targets, num_classes):
    smooth = 1
    confusion = confusion_matrix(preds, targets, num_classes)
    tp = torch.diagonal(confusion, dim1=-2, dim2=-1)
    fp = torch.sum(confusion, dim=-1) - tp
    fn = torch.sum(confusion, dim=-2) - tp

    dice_score = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    
    return dice_score

def mDice_per_class(preds, targets, num_classes):
    dice = dice_per_class(preds, targets, num_classes)
    dice = torch.mean(dice.reshape(-1, dice.shape[-1]), dim=0)
    
    return dice

def fDice(preds, targets, num_classes):
    confusion = confusion_matrix(preds, targets, num_classes)
    dice = dice_per_class(preds, targets, num_classes)

    
    pixel_per_class = torch.sum(confusion, dim=-1)
    total_pixels = torch.sum(pixel_per_class, dim=(-1,), keepdim=True)
    pixel_frequency = pixel_per_class / total_pixels
    
    
    
    fdice = torch.mean(torch.sum(pixel_frequency*dice, dim=-1).flatten()).item()
    
    return fdice

def mDice(preds, targets, num_classes):
    dice = dice_per_class(preds, targets, num_classes)
    mdice = torch.mean(dice.flatten()).item()
    
    return mdice


#####################################################################################################
####################################### All Metrics #################################################
#####################################################################################################

def all_metrics(preds, targets, num_classes):
    
    miou = mIOU(preds, targets, num_classes)
    fiou = fIOU(preds, targets, num_classes)
    accuracy = pixel_accuracy(preds, targets, num_classes)
    mean = mean_accuracy(preds, targets, num_classes)
    mdice = mDice(preds, targets, num_classes)
    fdice = fDice(preds, targets, num_classes)
    
    return miou, fiou, accuracy, mean, mdice, fdice
    