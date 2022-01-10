import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Transbase, tdcnn, mdl, rfb_trans, trdcnn, res_trans, res_encoder, vit

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

def get_vit():
    return vit.ViT(image_size=(72,24), patch_size=(12,4), num_classes=23, dim=512, depth=12, heads=12, mlp_dim=2048, channels=2, dim_head=1024)
