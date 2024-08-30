import matplotlib.pyplot as plt
import torch

def ViewDistributions(Labels, Counts, Colors, Title):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    
    ax1.pie(Counts, labels = Labels, colors = Colors)
    ax1.set_title(Title)

    ax2.bar(Labels, Counts, colors = Colors)
    ax1.set_title(Title)
    
    plt.tight_layout()
    
    plt.show()
    
def CakeGraph(Labels, Distributions, Colors, Title):
    
    plt.pie(Distributions, labels = Labels, colors = Colors)
    plt.axis("equal")
    plt.title(Title)
    plt.show()

def ShowLabels(TR):
    
    for i in range(0, 10, 5):
        
        fig, axs = plt.subplots(1, 5, figsize = (20, 4))
        
        axs[0].imshow(list(filter(lambda x: x[1] == i, TR))[0][0].numpy()[0], cmap = "gray")
        axs[1].imshow(list(filter(lambda x: x[1] == i+1, TR))[0][0].numpy()[0], cmap = "gray")
        axs[2].imshow(list(filter(lambda x: x[1] == i+2, TR))[0][0].numpy()[0], cmap = "gray")
        axs[3].imshow(list(filter(lambda x: x[1] == i+3, TR))[0][0].numpy()[0], cmap = "gray")
        axs[4].imshow(list(filter(lambda x: x[1] == i+4, TR))[0][0].numpy()[0], cmap = "gray")  
        
        axs[0].set_title(f"Label {i}")
        axs[1].set_title(f"Label {i+1}")
        axs[2].set_title(f"Label {i+2}")
        axs[3].set_title(f"Label {i+3}")
        axs[4].set_title(f"Label {i+4}") 

    plt.axis("off")    
    plt.show()
    