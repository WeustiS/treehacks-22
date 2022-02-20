import torch

class CountPeopleHead(torch.nn.Module):
    def __init__(self, encoder):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.enc = encoder
        
        self.fc1 = torch.nn.Linear(512,1024)

        self.fc2 = torch.nn.Linear(1024,1024)

        self.fc3 = torch.nn.Linear(1024,256)

        self.fc4 = torch.nn.Linear(256,1)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(.5)
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        enc = self.enc(x)
        
        x = self.fc1(enc)
        x = self.relu(x)

        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)

        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        
        return (x, enc)