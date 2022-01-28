import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model

from . import Transbase, tdcnn, mdl, rfb_trans, trdcnn, res_trans, res_encoder, vit, vit_wo_patch, separatedcnn_vit, separatedcnn_vit_wopatch, pvt, h21, separatedcnn_pvt, perceiver_pytorch, perceiver_io

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
'h21model',
'sep_pvt']

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
def get_vit(image_size=(360,180), patch_size=(20,10), num_classes=23, dim=1024, depth=12, heads=12, mlp_dim=2048, channels=6, dim_head=1024):
    return vit.ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)

@register_model
def separated(n_layer = 3):
    return separatedcnn_vit.separated(n_layer = n_layer)

@register_model
def cnn_vit_wo_patch(n_layer = 3):
    return separatedcnn_vit_wopatch.separated_wopatch(n_layer = n_layer)

@register_model
def pyramid(name='pvt_large'):
    # 'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
    return pvt[name](img_size=(360, 180), patch_size=(20,20), in_chans=6, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4)

@register_model
def h21model():
    return h21.Model2D()

@register_model
def sep_pvt(n_layer = 3, img_size=(360, 180), patch_size=(20,20), in_chans=3, num_classes=23, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):

    return separatedcnn_pvt.separated(n_layer, img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios, qkv_bias, 
                qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths, sr_ratios, num_stages)

@register_model
def perceiver(num_freq_bands=None,
        depth=12,
        max_freq=None,
        input_channels = 2,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 23,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = False,
        self_per_cross_attn = 1,
        final_classifier_head = True):
    return perceiver_pytorch.Perceiver(num_freq_bands = num_freq_bands, max_freq=max_freq, depth=depth, input_channels=input_channels, input_axis=input_axis, num_latents=num_latents, latent_dim=latent_dim, 
                                    cross_heads=cross_heads, latent_heads=latent_heads, cross_dim_head=cross_dim_head, latent_dim_head=latent_dim_head, num_classes=num_classes, attn_dropout=attn_dropout, ff_dropout=ff_dropout, 
                                    weight_tie_layers=weight_tie_layers, fourier_encode_data=fourier_encode_data, self_per_cross_attn=self_per_cross_attn, final_classifier_head=final_classifier_head)