# Path: src/models/unet_model.py
from segmentation_models_pytorch import Unet

def get_unet_model(encoder_name="resnet34", in_channels=3, classes=1):
    """
    Returns a U-Net model with the specified encoder backbone.
    """
    return Unet(encoder_name=encoder_name, in_channels=in_channels, classes=classes, activation=None)
