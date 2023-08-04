# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine import Config


class VidSwinBert():

    def __init__(self, args):
        swin = self._load_swin(args)
        bert, bert_cfg, tokenizer = self._load_bert(args)

    def generate(self, input):
        pass

    def _load_swin(self, args):
        from ._swin3d import SwinTransformer3D

        if int(args.img_res) == 384:
            assert args.vidswin_size == 'large'
            config_path = 'src/modeling/video_swin/swin_%s_384_patch244_window81212_kinetics%s_22k.py' % (
                args.vidswin_size, args.kinetics)
            model_path = './models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth' % (
                args.vidswin_size, args.kinetics)
        else:
            # in the case that args.img_res == '224'
            config_path = 'src/modeling/video_swin/swin_%s_patch244_window877_kinetics%s_22k.py' % (
                args.vidswin_size, args.kinetics)
            model_path = './models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth' % (
                args.vidswin_size, args.kinetics)

        cfg = Config.fromfile(config_path)
        swin = SwinTransformer3D(
            pretrained=None,
            pretrained2d=False,
            patch_size=cfg.model['backbone']['patch_size'],
            in_chans=3,
            embed_dim=cfg.model['backbone']['embed_dim'],
            depths=cfg.model['backbone']['depths'],
            num_heads=cfg.model['backbone']['num_heads'],
            window_size=cfg.model['backbone']['window_size'],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=torch.nn.LayerNorm,
            patch_norm=cfg.model['backbone']['patch_norm'],
            frozen_stages=-1,
            use_checkpoint=False)
        swin = VideoSwinWrapper(swin)
        ckpt = torch.load(model_path, map_location='cpu')
        swin.load_state_dict(ckpt['state_dict'], strict=False)
        return swin

    def _load_bert(self, args):
        pass


class VideoSwinWrapper(torch.nn.Module):

    def __init__(self, args, cfg, backbone):
        super(self).__init__()
        self.backbone = backbone
        self.use_grid_feature = args.grid_feat

    def forward(self, x):
        x = self.backbone(x)
        return x
