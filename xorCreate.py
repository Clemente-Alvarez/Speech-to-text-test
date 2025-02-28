import torch
import torch.nn as nn
import torch.optim as optim

class XORGateNN(nn.Module):
  def __init__(self):
    super(XORGateNN,self).__init__()
    #ReLU en las capas ocultas y sigmoid en salida
    self.layer = nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,1),nn.Sigmoid())

  def forward(self,x):
    return self.layer(x)