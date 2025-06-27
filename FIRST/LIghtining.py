import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner import Tuner

import matplotlib.pyplot as plt
import seaborn as sns


class BasicLightModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor([1.7]), requires_grad=True)
        self.b00 = nn.Parameter(torch.tensor([-0.85]), requires_grad=True)
        self.w01 = nn.Parameter(torch.tensor([-40.8]), requires_grad=True)
        self.w10 = nn.Parameter(torch.tensor([12.6]), requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w11 = nn.Parameter(torch.tensor([2.7]), requires_grad=True)
        self.finalBias = nn.Parameter(torch.tensor([-16.0]), requires_grad=True)
        self.learning_rate = 0.01

    def forward(self, input):
        input_to_top_Relu = F.relu((input * self.w00) + self.b00) * self.w01
        input_to_bottom_Relu = F.relu((input * self.w10) + self.b10) * self.w11
        relu_to_output = F.relu(input_to_bottom_Relu + input_to_top_Relu + self.finalBias)
        return relu_to_output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        loss = F.mse_loss(output_i, label_i)
        self.log("train_loss", loss)
        return loss


# Data Preparation
input_dosage = torch.linspace(0, 1, 11).unsqueeze(1)
output_effectiveness = torch.tensor([0.5, 0.7, 1.1, 1.5, 2.2, 2.5, 2.7, 3.2, 3.4, 3.8, 4.0]).unsqueeze(1)

ds = TensorDataset(input_dosage, output_effectiveness)
dl = DataLoader(ds, batch_size=5)

# Model Initialization
model = BasicLightModule()

# Learning Rate Finder
trainer = .Trainer(max_epochs=36, enable_progress_bar=False)
lr_find_results = trainer.tuner.lr_find(model, train_dataloaders=dl, min_lr=0.001, max_lr=1.0, early_stop_threshold=None)
new_lr = lr_find_results.suggestion()
print(f"Suggested learning rate: {new_lr}")
