# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmlmtools.parsers import BaseParser
from mmlmtools.schema import ToolMeta
from mmlmtools.utils import get_new_file_path
from mmlmtools.utils.cache import load_or_build_object
from ..base import BaseTool


class AnythingToImage:

    def __init__(self, device, eco_mode):
        try:
            from diffusers import StableUnCLIPImg2ImgPipeline

            from .models.imagebind_model import imagebind_huge as ib
        except ImportError as e:
            raise ImportError(
                f'Failed to run the tool for {e}, please check if you have '
                'install `diffusers` correctly')

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-unclip',
            torch_dtype=torch.float16,
            variation='fp16')
        self.device = device
        self.eco_mode = eco_mode
        self.pipe = pipe
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.model = ib.imagebind_huge(pretrained=True)
        self.model.eval()
        if not self.eco_mode:
            self.pipe.to(device)
            self.model.to(device)


class AudioToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Audio',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from audio. '
        'like: generate a real image from audio, '
        'or generate a new image based on the given audio. '
        'It takes an {{{input:audio}}} as the input, and returns '
        'the generated {{{output:image}}}.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device, eco_mode=True)
        self.pipe = self._inferencer.pipe
        self.model = self._inferencer.model
        self.device = self._inferencer.device
        self.eco_mode = self._inferencer.eco_mode

    def apply(self, audio: str) -> str:
        from .models.imagebind_model import imagebind_huge as ib

        if self.eco_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        audio_paths = [audio]
        audio_data = ib.load_and_transform_audio_data(audio_paths, self.device)
        embeddings = self.model.forward({ib.ModalityType.AUDIO: audio_data})
        embeddings = embeddings[ib.ModalityType.AUDIO]
        images = self.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = get_new_file_path(audio_paths[0], 'AudioToImage')
        images[0].save(new_img_name)

        if self.eco_mode:
            self.pipe.to('cpu')
            self.model.to('cpu')

        return new_img_name


class ThermalToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Thermal Image',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from a thermal image. '
        'like: generate a real image from thermal image, '
        'or generate a new image based on the given thermal image. '
        'It takes an {{{input:image}}} as the input and returns '
        'the generated {{{output:image}}}.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device, eco_mode=True)
        self.pipe = self._inferencer.pipe
        self.model = self._inferencer.model
        self.device = self._inferencer.device
        self.eco_mode = self._inferencer.eco_mode

    def apply(self, thermal_path: str) -> str:
        from .models.imagebind_model import imagebind_huge as ib

        if self.eco_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        thermal_paths = [thermal_path]
        thermal_data = ib.load_and_transform_thermal_data(
            thermal_paths, self.device)
        embeddings = self.model.forward(
            {ib.ModalityType.THERMAL: thermal_data})
        embeddings = embeddings[ib.ModalityType.THERMAL]
        images = self.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = get_new_file_path(thermal_data[0], 'ThermalToImage')
        images[0].save(new_img_name)

        if self.eco_mode:
            self.pipe.to('cpu')
            self.model.to('cpu')

        return new_img_name


class AudioImageToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Image and Audio',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from image and audio. '
        'like: generate a real image from image and audio, '
        'or generate a new image based on the given image and audio. '
        'The input to this tool should be an {{{input:image}}} and '
        'a {{{input:audio}}}. '
        'It returns the generated {{{output:image}}}.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device, eco_mode=True)
        self.pipe = self._inferencer.pipe
        self.model = self._inferencer.model
        self.device = self._inferencer.device
        self.eco_mode = self._inferencer.eco_mode

    def apply(self, image_path: str, audio_path: str) -> str:
        from .models.imagebind_model import imagebind_huge as ib

        if self.eco_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        # process image data
        vision_data = ib.load_and_transform_vision_data([image_path],
                                                        self.device)
        embeddings = self.model.forward({
            ib.ModalityType.VISION: vision_data,
        },
                                        normalize=False)
        img_embeddings = embeddings[ib.ModalityType.VISION]

        # process audio data
        audio_data = ib.load_and_transform_audio_data([audio_path],
                                                      self.device)
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: audio_data,
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]

        embeddings = (img_embeddings + audio_embeddings) / 2
        images = self.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = get_new_file_path(audio_path, 'AudioImageToImage')
        images[0].save(new_img_name)

        if self.eco_mode:
            self.pipe.to('cpu')
            self.model.to('cpu')

        return new_img_name


class AudioTextToImage(BaseTool):
    DEFAULT_TOOLMETA = dict(
        name='Generate Image from Audio and Text',
        model=None,
        description='This is a useful tool '
        'when you want to  generate a real image from audio and text prompt. '
        "like: generate a real image from audio with user's prompt, "
        'or generate a new image based on the given image audio with '
        "user's description. "
        'The input to this tool should be a {{{input:audio}}} and '
        'a {{{input:text}}} as the prompt. '
        'It returns the generated {{{output:image}}}.')

    def __init__(self,
                 toolmeta: Optional[ToolMeta] = None,
                 parser: Optional[BaseParser] = None,
                 remote: bool = False,
                 device: str = 'cuda'):
        super().__init__(toolmeta, parser, remote, device)

    def setup(self):
        self._inferencer = load_or_build_object(
            AnythingToImage, device=self.device, eco_mode=True)
        self.pipe = self._inferencer.pipe
        self.model = self._inferencer.model
        self.device = self._inferencer.device
        self.eco_mode = self._inferencer.eco_mode

    def apply(self, audio_path: str, prompt: str) -> str:
        from .models.imagebind_model import imagebind_huge as ib

        if self.eco_mode:
            self.pipe.to(self.device)
            self.model.to(self.device)

        audio_paths = [audio_path]
        text = ib.load_and_transform_text([prompt], self.device)
        embeddings = self.model.forward({ib.ModalityType.TEXT: text},
                                        normalize=False)
        text_embeddings = embeddings[ib.ModalityType.TEXT]

        audio_data = ib.load_and_transform_audio_data(audio_paths, self.device)
        embeddings = self.model.forward({
            ib.ModalityType.AUDIO: audio_data,
        })
        audio_embeddings = embeddings[ib.ModalityType.AUDIO]
        embeddings = text_embeddings * 0.5 + audio_embeddings * 0.5
        images = self.pipe(
            image_embeds=embeddings.half(), width=512, height=512).images
        new_img_name = get_new_file_path(audio_paths[0], 'AudioTextToImage')
        images[0].save(new_img_name)

        if self.eco_mode:
            self.pipe.to('cpu')
            self.model.to('cpu')

        return new_img_name