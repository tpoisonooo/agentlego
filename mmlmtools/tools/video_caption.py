# Copyright (c) OpenMMLab. All rights reserved.
# This file is adapted from
# https://github.com/yukw777/VideoBLIP/blob/main/demo/app.py
# by Yu Keunwoo Peter under the MIT license

from typing import Optional

from mmlmtools.utils.toolmeta import ToolMeta
from .base_tool import BaseTool
from .parsers import BaseParser

try:
    from transformers import Blip2Processor

    from .video_blip import VideoBlipForConditionalGeneration, process
    has_video_blip = True
except ImportError as e:
    has_video_blip = False
    video_blip_import_error = e

try:
    from pytorchvideo.data.video import VideoPathHandler
    has_pytorchvideo = True
except ImportError:
    has_pytorchvideo = False


class VideoCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Video Caption',
        model='kpyu/video-blip-flan-t5-xl-ego4d',
        description='This is a useful tool when you want to generate '
        'description for a video. It takes a {{{input:video}}} as the input, '
        'and returns a {{{output:text}}} representing the description of the '
        'video. ')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 num_beams: int = 4,
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 remote: bool = False,
                 device: str = 'cpu'):

        super().__init__(toolmeta, parser, remote, device)

        if remote:
            raise NotImplementedError('`VideoCaptionTool` does not support '
                                      'remote mode.')
        if not has_video_blip:
            raise ImportError(
                'Required module transformers or video_blip are not imported '
                'successfully. Please refer to the follow error message: '
                f'{video_blip_import_error}')

        if not has_pytorchvideo:
            raise ImportError('Please install pytorchvideo first.')

        self.num_beams = num_beams
        self.maxnewtokens = max_new_tokens
        self.temperature = temperature
        self.video_path_handler = VideoPathHandler()

    def setup(self):

        self.processor = Blip2Processor.from_pretrained(self.toolmeta.model)
        self.model = VideoBlipForConditionalGeneration.from_pretrained(
            self.toolmeta.model).to(self.device).eval()

    def apply(self, video_path: str) -> str:
        # process only the first 10 seconds
        video_path = video_path.strip()
        clip = self.video_path_handler(video_path).get_clip(0, 10)

        # sample a frame every 30 frames, i.e. 1 fps.
        # We assume the video is 30 fps for now.
        frames = clip['video'][:, ::30].unsqueeze(0)

        # construct chat context
        context = ''
        inputs = process(self.processor, frames, text=context).to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            num_beams=self.num_beams,
            max_new_tokens=self.maxnewtokens,
            temperature=self.temperature)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text
