import torch
import torch.nn as nn
import torch.optim as optim

from xorCreate import XORGateNN


#load model
try:
    model = XORGateNN()
    model.load_state_dict(torch.load("weights.pt"))
    model.eval()
except FileNotFoundError:
    print("Error weights.pt file does not exists")
    exit()

print("Input test: ")
while True:
    try:
        a,b = map(int,input().split(" "))
        print("Predicted: ")
        X = torch.tensor([[a,b]],dtype=torch.float32)
        print(model.forward(X))
    except ValueError:
        print("Incorrect input try to put a space between the numbers")