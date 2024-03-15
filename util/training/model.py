import lightning as L
from torch.optim import AdamW


class BasicModel(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["input"])
        loss = self._calculate_loss(outputs, batch["label"], batch["mask"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch["input"])
        loss = self._calculate_loss(outputs, batch["label"], batch["mask"])
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1
        )

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch["input"])
        loss = self._calculate_loss(outputs, batch["label"], batch["mask"])
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

    def _calculate_loss(self, outputs, label, mask):
        return (outputs[mask] - label[mask]).pow(2).mean()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-3,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
        )
        return optimizer
