# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torchvision import transforms

from mmlmtools.utils.toolmeta import ToolMeta
from ..base_tool import BaseTool
from ..parsers import BaseParser
from .utils.load import LoadVideo
from .utils.tag2text import tag2text_caption


class VideoCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Video Description',
        model={'model': 'model_zoo/tag2text_swin_14m.pth'},
        description='This is a useful tool when you want to generate '
        'description for a video. It takes a {{{input:video}}} as the input, '
        'and returns a {{{output:text}}} representing the description of the '
        'video. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cpu'):
        super().__init__(toolmeta, parser, remote, device)
        if remote:
            raise NotImplementedError('`VideoCaptionTool` does not support '
                                      'remote mode.')

        self.frame_size = 384
        self.transform = None
        self.model = None
        self.load_video = None

    def setup(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = tag2text_caption(
            pretrained=self.toolmeta.model['model'],
            image_size=self.frame_size,
            vit='swin_b').eval().to(self.device)
        self.load_video = LoadVideo()

    def apply(self, video_path: str) -> str:

        video_path = video_path.strip()
        vid = self.load_video(video_path)
        tmp = []
        for _, frame in enumerate(vid):
            tmp.append(self.transform(frame).to(self.device).unsqueeze(0))

        data = torch.cat(tmp)
        input_tag_list = None
        captions = self.model.generate(
            data, input_tag_list, max_length=50, return_tag_predict=False)

        del vid, data, tmp
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        caption = '. '.join(captions)

        return caption
