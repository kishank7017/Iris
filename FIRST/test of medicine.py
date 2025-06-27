import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Neural network
class BasicNNtrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.b00 = nn.Parameter(torch.tensor([0.0]),requires_grad=True)
        self.w01 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.b01 = nn.Parameter(torch.tensor([0.0]),requires_grad=True)
        self.w10 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor([0.0]),requires_grad=True)
        self.w11 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.b11 = nn.Parameter(torch.tensor([0.0]),requires_grad=True)
        self.w02 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.w12 = nn.Parameter(torch.tensor([0.01]),requires_grad=True)
        self.b23 = nn.Parameter(torch.tensor([0.0]),requires_grad=True)
        self.lr = nn.Parameter(torch.tensor([0.08]),requires_grad=True)

    def forward(self, x):  # x shape: [1, 2]
        input0 = x[:, 0]
        input1 = x[:, 1]

        h0 = torch.tanh(input0 * self.w00 + self.b00 + input1 * self.w10 + self.b10)
        h1 = torch.tanh(input0 * self.w01 + self.b01 + input1 * self.w11 + self.b11)
        output = h0 * self.w02 + h1 * self.w12 + self.b23
        return output.unsqueeze(1)  # shape: [1, 1]

# Training data
X = torch.tensor([
    [-2.0, -2.0],
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 0.0,  0.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [ 2.0,  2.0],
    [ 0.5,  0.5],
    [-0.5, -0.5],
    [ 2.0, -2.0],
], dtype=torch.float32)

Y = torch.tensor([
    [-0.9993],
    [-0.7616],
    [  0.0   ],
    [  0.0   ],
    [  0.0   ],
    [  0.7616],
    [  0.9993],
    [ 0.4621],
    [-0.4621],
    [  0.0   ],
], dtype=torch.float32)

# Model and optimizer
model = BasicNNtrain()
optimizer = SGD(model.parameters(), model.lr)

# Training loop (your style)
losses = []

for epoch in range(1000):
    total_loss = 0.0

    for i in range(len(X)):
        x = X[i].unsqueeze(0)  # shape: [1, 2]
        y = Y[i].unsqueeze(0)  # shape: [1, 1]

        output = model(x)
        loss = F.mse_loss(output, y)

        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    losses.append(total_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.6f} | Bias: {model.b23.item():.4f}")

    if total_loss < 1e-5:
        print(f"\nEarly stop at epoch {epoch} with loss {total_loss:.6f}")
        break

# Plot predictions vs true values
with torch.no_grad():
    predictions = model(X).squeeze().numpy()
    true_values = Y.squeeze().numpy()

plt.figure(figsize=(8, 5))
plt.title("True vs Predicted Outputs")
sns.scatterplot(x=true_values, y=predictions, color="blue", s=80)
plt.plot([-1, 1], [-1, 1], linestyle='--', color='gray')
plt.xlabel("True Output")
plt.ylabel("Predicted Output")
plt.grid(True)
plt.show()
print("\nTrained Weights and Biases:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item():.4f}")
