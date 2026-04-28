import sys
sys.path.append(".")
sys.path.append("../../")

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import lightning as L
import torchmetrics
import torch.nn.functional as F
from lightning.pytorch import seed_everything
seed_everything(42, workers=True)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from .embedding_dataset import Embedding_dataset, Embedding_dataset_embdi
import pandas as pd

class Embedding_scaler(L.LightningModule):
    def __init__(self, input_size: int, output_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(input_size, output_size)
        self.lr = lr
        self.loss = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        e_r__, e_s__, overlap_ratio = batch
        with torch.autocast(device_type='cuda', enabled=False):
            e_r = self.forward(e_r__.squeeze())
            e_s = self.forward(e_s__.squeeze())
            pred = F.cosine_similarity(e_r, e_s)
            try:
                loss = self.loss(pred, overlap_ratio)
            except RuntimeError:
                pred = pred.reshape((-1))
                loss = self.loss(pred, overlap_ratio)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        e_r__, e_s__, overlap_ratio = batch
        with torch.autocast(device_type='cuda', enabled=False):
            e_r = self.forward(e_r__.squeeze())
            e_s = self.forward(e_s__.squeeze())
            pred = F.cosine_similarity(e_r, e_s)
            try:
                loss = self.loss(pred, overlap_ratio)
            except RuntimeError:
                pred = pred.reshape((-1))
                loss = self.loss(pred, overlap_ratio)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        e_r__, e_s__, overlap_ratio = batch
        with torch.autocast(device_type='cuda', enabled=False):
            e_r = self.forward(e_r__.squeeze())
            e_s = self.forward(e_s__.squeeze())
            try:
                pred = F.cosine_similarity(e_r, e_s)
            except:
                pred = F.cosine_similarity(e_r.unsqueeze(0), e_s.unsqueeze(0))
            try:
                loss = self.loss(pred, overlap_ratio)
            except RuntimeError:
                pred = pred.reshape((-1))
                loss = self.loss(pred, overlap_ratio)
        self.log('test_loss', loss)
        return loss

trainer_config =   {
    'accelerator': "gpu",
    'fast_dev_run' : False, 
    'deterministic' : False, 
    'devices':[0],
    'min_epochs':1,
    'max_epochs':10,
    'log_every_n_steps':50,
  }

if __name__ == '__main__':
    data_train = [(torch.rand(100), torch.rand(100)), (torch.rand(100),torch.rand(100)), (torch.rand(100), torch.rand(100)), (torch.rand(100), torch.rand(100)), (torch.rand(100), torch.rand(100))]
    data_test = [(torch.rand(100), torch.rand(100)), (torch.rand(100),torch.rand(100))]
    data_valid = [(torch.rand(100), torch.rand(100)), (torch.rand(100),torch.rand(100))]
    dataset_train = DataLoader(Embedding_dataset(data_train), num_workers=0, batch_size=2)
    dataset_test = DataLoader(Embedding_dataset(data_test), num_workers=0, batch_size=2)
    dataset_valid = DataLoader(Embedding_dataset(data_valid), num_workers=0, batch_size=2)


    model = Embedding_scaler()
    callbacks = []
    trainer = L.Trainer(**trainer_config, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=dataset_train, val_dataloaders=dataset_valid)
    trainer.test(model, dataloaders=dataset_test)