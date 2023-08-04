# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmlmtools.tools.base_tool import BaseTool
from mmlmtools.tools.parsers import BaseParser
from mmlmtools.utils.toolmeta import ToolMeta


class VideoCaptionTool(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Video Caption',
        model='swin-bert',
        description='This is a useful tool when you want to generate '
        'description for a video. It takes a {{{input:video}}} as the input, '
        'and returns a {{{output:text}}} representing the description of the '
        'video. ')

    def __init__(
            self,
            toolmeta: Optional[ToolMeta] = None,
            parser: Optional[BaseParser] = None,
            #  num_beams: int = 4,
            #  max_new_tokens: int = 128,
            #  temperature: float = 0.7,
            remote: bool = False,
            device: str = 'cpu'):

        super().__init__(toolmeta, parser, remote, device)

        if remote:
            raise NotImplementedError('`VideoCaptionTool` does not support '
                                      'remote mode.')
