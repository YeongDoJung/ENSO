import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model

from . import Transbase, tdcnn, mdl, rfb_trans, trdcnn, res_trans, res_encoder, vit, vit_wo_patch, separatedcnn_vit, separatedcnn_vit_wopatch, pvt, h21

__all__ = ['Model_2D',
'encoders',
'Transformer',
'Model_3D',
'res_trf',
'res_encs',
'get_vit',
'separated',
'cnn_vit_wo_patch',
'pyramid',
'h21model',]

@register_model
def Model_2D():
    return tdcnn.Model2D()

@register_model
def encoders(in_channel=2, out_channel=16):
    return mdl.RFB_Transformer(in_channel, out_channel)

@register_model
def Transformer(in_channel=2, out_channel=16):
    return rfb_trans.RFB_Trf(in_channel, out_channel)

@register_model
def Model_3D():
    return trdcnn.Model3D(2,16,256)

@register_model
def res_trf():
    return res_trans.res_transformer()

@register_model
def res_encs():
    return res_encoder.res_enc()

@register_model
def get_vit(image_size=(360,180), patch_size=(12,4), num_classes=23, dim=1024, depth=12, heads=12, mlp_dim=2048, channels=6, dim_head=1024):
    return vit.ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)

@register_model
def separated(n_layer = 3):
    return separatedcnn_vit.separated(n_layer = n_layer)

@register_model
def cnn_vit_wo_patch(n_layer = 3):
    return separatedcnn_vit_wopatch.separated_wopatch(n_layer = n_layer)

@register_model
def pyramid(name):
    # 'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
    return pvt[name]

@register_model
def h21model():
    return h21.Model2D()