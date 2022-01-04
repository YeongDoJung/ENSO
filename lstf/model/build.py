import torch
import torch.nn as nn
import torch.nn.functional as F

from lstf.model import mdl, tdcnn, trdcnn, rfb_trans

__all__ = ['encoders', 'decoders', 'Model_3D', 'Model_2D']


def Model_2D():
    return tdcnn.Model2D()

def encoders(in_channel, out_channel):
    return mdl.RFB_Transformer(in_channel, out_channel)

def Transformer(in_channel, out_channel):
    return rfb_trans.RFB_Trf(in_channel, out_channel)

def Model_3D():
    return trdcnn.Model3D(2,16,256)