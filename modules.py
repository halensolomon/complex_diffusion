import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tensorflow import tensorboard
import lightning as pl

# Define complex-valued autoencoder
class complexAutoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
        
    def forward(self, x):
        #
        x, y = batch
        y_hate
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
class complexReLu(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x):
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class modReLu(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x):
        if self.bias + x.abs() > 0:
            return self.bias + x.abs()
        else
            return 0
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
class complexTanH(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x):
        return torch.tanh(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class attention(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
        
    def forward(self, x):
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
class complexDownSample(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x, window_size = 2, stride = 2):
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class complexUpSample(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x):
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
class complexRelu(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)
    def forward(self, x):
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)