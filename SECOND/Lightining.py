import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.tuner.tuning import Tuner

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.Tensor([1.7]), requires_grad=False)
        self.b00 = nn.Parameter(torch.Tensor([-0.85]), requires_grad=False)
        self.w01 = nn.Parameter(torch.Tensor([-40.8]), requires_grad=False)

        self.w10 = nn.Parameter(torch.Tensor([12.6]), requires_grad=False)
        self.b10 = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.w11 = nn.Parameter(torch.Tensor([2.7]), requires_grad=False)

        self.finalBias = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

        self.learning_rate = 0.01

    def forward(self, input):
        input_to_top_Relu = F.relu((input * self.w00) + self.b00) * self.w01
        input_to_bottom_Relu = F.relu((input * self.w10) + self.b10) * self.w11
        relu_to_output = F.relu(input_to_bottom_Relu + input_to_top_Relu + self.finalBias)
        return relu_to_output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        inputi, labeli = batch
        outputi = self.forward(inputi)
        loss = F.mse_loss(outputi, labeli)
        self.log("train_loss", loss)
        return loss

model=BasicLightningTrain()

# Data preparation
input = torch.tensor([0.0, 0.5, 1.0]).unsqueeze(1)
label = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(1)
dataset = TensorDataset(input, label)
dataloader = DataLoader(dataset, batch_size=1)

# Trainer
trainer = Trainer(max_epochs=40, logger=False, enable_model_summary=False)

# Tuner with lr_find
tuner = Tuner(trainer)
lr_find_results = tuner.lr_find(model=model, train_dataloaders=dataloader, min_lr=0.01, max_lr=1.0)

# Update model learning rate
new_lr = lr_find_results.suggestion()
print(f"Suggested learning rate: {new_lr}")
model.learning_rate = new_lr

# Train
trainer.fit(model, train_dataloaders=dataloader)

# Plotting
input_trial = torch.linspace(0, 1, 11).unsqueeze(1)
output_trial = model(input_trial).detach()

sns.set_style(style="whitegrid")
sns.lineplot(x=input_trial.squeeze(), y=output_trial.squeeze(), color='green', linewidth=2.5)
plt.ylabel("effectiveness")
plt.xlabel("dosage")
plt.title("hello world")
plt.show()
