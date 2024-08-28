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
    
    
def plot_mnist(train, width=10, height=5, cmap="gray"):
    
    print(train[0])
    
    imgs = []
    for i in range(10):
        for img, label in train:
            if label == torch.tensor(i):
                imgs.append(img)
                break

    
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(10):
        
        ax = axs[i // 5, i % 5]
        
        ax.imshow(imgs[i], cmap = "gray")
        
        ax.set_title(f"Class {i}")
        ax.axis("off")

    fig.suptitle("MNIST dataset")
    fig.tight_layout()
    


if __name__ == "__main__" : 
    
    Tensor1 = torch.tensor([1, 2, 3, 4])
    Tensor2 = torch.tensor([5, 6, 7])

    Tensor3 = torch.tensor([])

    print(Tensor3.detach().numpy())

    print(len(Tensor3.detach().numpy()))

    TimeSeries1 = [2, 3, 5, 4, 1, 6]

    TimeSeries2 = [1, 2, 3, 5]

    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 6))

    plt.plot(range(len(TimeSeries1)), TimeSeries1, label = "Sequence 1", color = "r")
    plt.plot(range(2, len(TimeSeries2)+2), TimeSeries2, label = "Sequence 2", color = "b")

    plt.legend()

    plt.show()