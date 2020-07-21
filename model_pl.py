#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from transforms import *
from criteria import *


class PLWrapper(pl.LightningModule):
    
    def __init__(self, model, criterion, dl_train, dl_val, plot_loss=True):
        
        super(pl.LightningModule, self).__init__()
        
        self.model = model
        self.criterion = criterion
        self.dl_train = dl_train
        self.dl_val = dl_val
        
        self.example_input_array = None
        
        self.plot_loss = plot_loss
        if plot_loss == True:
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax.set_ylim(0,1)
            self.ax.legend(['Train','Val'])
            self.hl_train, = self.ax.plot([], [])
            self.hl_val, = self.ax.plot([], [])
            self.fig.canvas.draw()
            
        
    def forward(self, x):
        return self.model(x)
    
    
    def train_dataloader(self):
        return self.dl_train
    
    
    def val_dataloader(self):
        return self.dl_val
    
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        return {'loss': loss, 
                'progress_bar': {'training_loss': loss},
                'log': {'training_loss': loss}}
    
    
    def training_epoch_end(self, outputs):
        if self.plot_loss:
            train_loss_epoch = np.array([e['loss'].item() for e in outputs]).mean()
            xdata, ydata = self.hl_train.get_xdata(), self.hl_train.get_ydata()
            self.hl_train.set_xdata(np.append(xdata, xdata[-1]+1 if len(xdata)>0 else 1))
            self.hl_train.set_ydata(np.append(ydata, np.mean(train_loss_epoch)))
            self.ax.relim(); self.ax.autoscale(axis='x'); self.fig.canvas.draw()
        return {"loss": torch.stack([e['loss'] for e in outputs]).mean()}
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        # calculate more metrics here
        return {'val_loss': loss, 
                'progress_bar': {'validation_loss': loss},
                'log': {'validation_loss': loss}}
    
    
    def validation_epoch_end(self, outputs):
        if self.plot_loss:
            val_loss_epoch = np.array([e['val_loss'].item() for e in outputs]).mean()
            xdata, ydata = self.hl_val.get_xdata(), self.hl_val.get_ydata()
            self.hl_val.set_xdata(np.append(xdata, xdata[-1]+1 if len(xdata)>0 else 1))
            self.hl_val.set_ydata(np.append(ydata, np.mean(val_loss_epoch)))
            self.ax.relim(); self.ax.autoscale(axis='x'); self.fig.canvas.draw()
        return {"val_loss": torch.stack([e['val_loss'] for e in outputs]).mean()}  # for checkpoint / early stopping
