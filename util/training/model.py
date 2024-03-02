import lightning as L
import torch.nn as nn
from torch.optim import AdamW


class BasicModel(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.99,
        epsilon: float = 1e-8,
        loss=nn.MSELoss(),
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def forward(self, batch):
        position = self.model(batch["input"])
        loss = self.loss(position, batch["label"])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-3,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
        )
        return optimizer
