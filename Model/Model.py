import torch
import torch.nn as nn

class ModelCNN(nn.Module):
    
    def __init__(self):
        
        super(ModelCNN, self).__init__()
        
        self.__ClassList = []     #Lista contenenti le classi
        self.__FinalNeurons = []  #Lista contenente i layer con un solo neurone a testa (ogni layer rappresenta una classe).
        
        #Sequenza di layer convoluzionali
        self.__LayerCNN1 = nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 7, stride = 2)
        self.__LayerCNN2 = nn.Conv2d(in_channels = 5, out_channels = 3, kernel_size = 5, stride = 1)
        self.__LayerCNN3 = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, stride = 1)
        
        #Sequenza di layer densamente conessi
        self.__Fully1 = nn.Linear(25, out_features = 20)
        self.__Fully2 = nn.Linear(20, out_features = 15)
        self.__Fully3 = nn.Linear(15, out_features = 10)
        
        #Layer di softmax
        self.__Softmax = nn.Softmax(dim = 1)
    
    #Metodo di aggiunta di una nuova classe 
    def AddClass(self, Class):
        
        #Se la classe Class non è stata ancora incontrata dal modello, la aggiungo, con il suo layer (con un neurone) associato.
        if(Class not in self.__ClassList):
            self.__ClassList.append(Class)
            self.__FinalNeurons.append(nn.Linear(10, 1))
    
    #Operazione di calcolo
    def forward(self, x):
        
        print(x.size())
        
        #Fase di passaggio nella convoluzione
        x = self.__LayerCNN1(x)
        x = self.__LayerCNN2(x)
        x = self.__LayerCNN3(x)
        
        print(x.size())
        
        #Appiattimento dell' immagine in un vettore
        x = x.view(x.size(0), -1)
        
        print(x.size())
        
        #Fase di passaggio negli srati densamente connessi
        x = self.__Fully1(x)
        x = self.__Fully2(x)
        x = self.__Fully3(x)
        
        print(x.size())
        
        print(tuple([FinalNeuron(x) for FinalNeuron in self.__FinalNeurons]))
        
        #Calcolo dei valori su ogni neurone, dove ogni neurone rappresenta una classe
        Results = torch.cat(tuple([FinalNeuron(x) for FinalNeuron in self.__FinalNeurons]), dim = 1)
        
        #Applicazione della softmax
        Results = self.__Softmax(Results)
        
        #Calcolo dell' indice della classe più probabile per ogni dato del batch
        return torch.argmax(Results, dim = 1)