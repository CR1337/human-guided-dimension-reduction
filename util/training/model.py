import lightning as L
import torch.nn as nn

class BasicModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3, loss=nn.MSELoss()):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss

    def forward(self, batch):
        position = self.model(batch["x"])
        loss = self.loss(position, batch["y"])
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        return loss
    
    def configure_optimizers(self):
        optimizer = nn.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer