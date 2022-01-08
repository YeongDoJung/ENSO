import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tdcnn, mdl, rfb_trans, trdcnn, res_trans, res_encoder

def Model_2D():
    return tdcnn.Model2D()

def encoders(in_channel, out_channel):
    return mdl.RFB_Transformer(in_channel, out_channel)

def Transformer(in_channel, out_channel):
    return rfb_trans.RFB_Trf(in_channel, out_channel)

def Model_3D():
    return trdcnn.Model3D(2,16,256)

def res_trf():
    return res_trans.res_transformer()

def res_encs():
    return res_encoder.res_enc()
