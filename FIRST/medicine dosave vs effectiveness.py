import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class BasicNNtrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.Tensor([1.7]), requires_grad=False)
        self.b00 = nn.Parameter(torch.Tensor([-0.85]), requires_grad=False)
        self.w01 = nn.Parameter(torch.Tensor([-40.8]), requires_grad=False)
        self.w10 = nn.Parameter(torch.Tensor([12.6]), requires_grad=False)
        self.b10 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.w11 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.finalBias = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, input):
        input_to_top_Relu = F.relu((input * self.w00) + self.b00) * self.w01
        input_to_bottom_Relu = F.relu((input * self.w10) + self.b10) * self.w11
        relu_to_output = F.relu(input_to_bottom_Relu + input_to_top_Relu + self.finalBias)
        return relu_to_output
    
    

# # Inputs and model
input_of_nn = torch.linspace(0, 1, 11)
model = BasicNNtrain()
output_values_of_nn = model(input_of_nn)
# print(output_values_of_nn)

# Training data
# inputs = torch.tensor([0.0, 0.5, 0.0])
# label = torch.tensor([0.0, 1.0, 0.0])
inputs = torch.tensor([0.0000, 0.6981, 1.3963, 2.0944, 2.7925, 3.4907, 4.1888, 4.8869, 5.5851, 6.2832])
label = torch.tensor([ 0.0000,  0.6428,  0.9848,  0.8660,  0.3420, -0.3420, -0.8660, -0.9848, -0.6428, -0.0000])
optimizer = SGD(model.parameters(), lr=0.1)

print("Initial value before gradient descent: " + str(model.finalBias.data) + "\n")

# Training loop
for epoch in range(100):
    total_loss = 0

    for iteration in range(len(inputs)):
        input_tensor = torch.tensor([inputs[iteration]])
        loss = (model(input_tensor) - label[iteration]) ** 2
        loss.backward()
        total_loss += float(loss)

    if total_loss < 0.0001:
        print("num steps: " + str(epoch))
        break

    optimizer.step()
    optimizer.zero_grad()
    print("steps: " + str(epoch) + " final bias: " + str(model.finalBias.data))

input_of_n = torch.linspace(0, 1, 11)
output_values_of_n = model(input_of_n)
print(output_values_of_n)

# Plot after training
output_values_of_nn = model(input_of_nn)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_of_nn.detach(), y=output_values_of_nn.detach(), color="green", linewidth=2.5)
plt.ylabel("effectiveness")
plt.xlabel("dosage")
plt.show()
