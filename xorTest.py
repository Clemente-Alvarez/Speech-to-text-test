import torch
import torch.nn as nn
import torch.optim as optim

from xorTrain import XORGateNN


model = XORGateNN()

#load model
model.load_state_dict("weights.pt")
model.eval()

while True:
    a,b = map(int,input().split(" "))
    print("Predicted: ")
    X = torch.tensor([[a,b]],dtype=torch.float32)
    print(model.forward(X))