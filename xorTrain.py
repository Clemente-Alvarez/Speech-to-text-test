import torch
import torch.nn as nn
import torch.optim as optim

#Inputs
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32) #Inputs
Y = torch.tensor([[0],[1],[1],[0]],dtype=torch.float32) # Outputs

class XORGateNN(nn.Module):
  def __init__(self):
    super(XORGateNN,self).__init__()
    #ReLU en las capas ocultas y sigmoid en salida
    self.layer = nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,1),nn.Sigmoid())

  def forward(self,x):
    return self.layer(x)


model = XORGateNN()
criterion = nn.BCELoss() # Binary cross entropy. Se puede usar el Mean Square Error
optimizer = optim.SGD(model.parameters(), lr=0.1) #Usa stachastic gradient descent (el clasico) con tasa_aprende=0.1

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
#safe model
torch.save(model.state_dict(), "weights.pt")