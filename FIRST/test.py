import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class BasicNNtrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Set initial values manually, but allow training
        self.w00 = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=True)
        self.b00 = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=True)
        self.w01 = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=True)
        self.w10 = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)
        self.w11 = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=True)
        self.finalBias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)

    def forward(self, input):
        input = input.view(-1)
        input_to_top_Relu = F.relu((input * self.w00) + self.b00) * self.w01
        input_to_bottom_Relu = F.relu((input * self.w10) + self.b10) * self.w11
        output = F.relu(input_to_bottom_Relu + input_to_top_Relu + self.finalBias)
        return output

# Inputs and model
input_of_nn = torch.linspace(0, 1, 11)
model = BasicNNtrain()
print("Initial output before training:", model(input_of_nn).detach())

# Training data
inputs = torch.tensor([0.0, 0.5, 0.0])
labels = torch.tensor([1.0, 0, 1.0])
optimizer = SGD(model.parameters(), lr=0.1)

print("\nInitial finalBias value:", model.finalBias.data, "\n")

# Training loop
for epoch in range(100):
    total_loss = 0.0

    for i in range(len(inputs)):
        input_tensor = inputs[i].unsqueeze(0)
        target = labels[i]

        output = model(input_tensor)
        loss = (output - target) ** 2
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch}: Loss = {total_loss:.6f}, Final Bias = {model.finalBias.item():.4f}")

    if total_loss < 0.0001:
        print("Converged at epoch", epoch)
        break

# Evaluate after training
input_of_n = torch.linspace(0, 1, 20)
output_values_of_n = model(input_of_n).detach()
print("Output after training:", output_values_of_n)

# Plot after training
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_of_nn.detach(), y=model(input_of_nn).detach(), color="green", linewidth=2.5)
plt.ylabel("Effectiveness")
plt.xlabel("Dosage")
plt.title("Neural Network Output After Training")
plt.show()
