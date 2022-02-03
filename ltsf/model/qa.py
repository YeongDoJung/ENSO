from mimetypes import init
import torch
import torch.nn as nn
import torch.functional as F

class spatialattention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        