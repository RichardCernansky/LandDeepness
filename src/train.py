# Path: src/train.py
import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.datamodules.dataset import SegmentationDataset
from src.models.unet_model import get_unet_model
from src.losses.loss import get_loss_function
from src.metrics.metrics import iou_metric
from src.utils.helpers import save_model

class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss_fn, learning_rate):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        iou = iou_metric(preds, masks)
        self.log("val_loss", loss)
        self.log("val_iou", iou)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def main():
    # Load config
    with open("configs/config_segment.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Prepare data
    train_dataset = SegmentationDataset(
        config["data"]["train_images"],
        config["data"]["train_masks"],
        config["training"]["img_size"]
    )
    val_dataset = SegmentationDataset(
        config["data"]["val_images"],
        config["data"]["val_masks"],
        config["training"]["img_size"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # Prepare model, loss, and trainer
    model = get_unet_model(encoder_name=config["training"]["model_name"])
    loss_fn = get_loss_function()
    seg_model = SegmentationModel(model, loss_fn, config["training"]["learning_rate"])

    trainer = pl.Trainer(max_epochs=config["training"]["epochs"], accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(seg_model, train_loader, val_loader)

    # Save model
    save_model(seg_model.model, config["output"]["model_path"])

if __name__ == "__main__":
    main()
