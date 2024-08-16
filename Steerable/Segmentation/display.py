import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def display_segmentation_map(target_data, num_classes=None, fig = None, ax=None, colorbar=True):
    
    if num_classes is None:
        num_classes = torch.max(torch.unique(target_data)).item() + 1

    colors = [plt.get_cmap(cmap)(i) for cmap in ['Set1', 'Set2', 'Set3'] for i in range(plt.get_cmap(cmap).N)]
    #colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    colors = ['black'] + colors[:num_classes-1]
    cmap = ListedColormap(colors)
    

    boundaries = torch.arange(-1.5, num_classes - 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)
    if ax is None:
        plt.imshow(target_data - 1, cmap=cmap, norm=norm)
        plt.axis('off')
        if colorbar:
            plt.colorbar(boundaries=torch.arange(0, num_classes-1), ticks=torch.arange(0, num_classes-1))
        
    else:
        ax.axis('off')
        cax = ax.imshow(target_data - 1, cmap=cmap, norm=norm)
        if colorbar:
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(cax, boundaries=torch.arange(0, num_classes-1), ticks=torch.arange(0, num_classes-1), cax=cbar_ax)
        

    return

def display_prob_map(prob_data, fig = None, ax=None):
    
    cmap = plt.cm.get_cmap('viridis') 
    cmap = cmap.reversed() 
    
    if ax is None:
        plt.imshow(prob_data - 1, cmap=cmap)
        plt.colorbar(boundaries=torch.arange(0, 10), ticks=torch.arange(0, 10))
        plt.axis('off')
        plt.show()
    else:
        cax = ax.imshow(prob_data, cmap=cmap)
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(cax, cax=cbar_ax)

    return