import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Transbase, tdcnn, mdl, rfb_trans, trdcnn, res_trans, res_encoder, vit, vit_wo_patch, separatedcnn_vit, separatedcnn_vit_wopatch, pvt

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

def get_vit(image_size=(72,24), patch_size=(12,4), num_classes=23, dim=512, depth=12, heads=12, mlp_dim=2048, channels=2, dim_head=1024):
    return vit.ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)

def separated(n_layer = 3):
    return separatedcnn_vit.separated(n_layer = n_layer)

def cnn_vit_wo_patch(n_layer = 3):
    return separatedcnn_vit_wopatch.separated_wopatch(n_layer = n_layer)

def pyramid(name):
    # 'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
    return pvt[name]