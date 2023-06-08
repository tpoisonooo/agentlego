# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from collections import defaultdict
from pickle import dumps
from typing import Optional, Union

import mmlmtools.tools as tools
from .toolmeta import ToolMeta
from .tools.base_tool import BaseTool

# Loaded from OpenMMLab metafiles, the loaded DEFAULT_TOOLS will be like this:

# DEFAULT_TOOLS = {
#     'ImageCaptionTool':
#     dict(
#         model='blip-base_3rdparty_caption',
#         description=
#         'useful when you want to know what is inside the photo. receives image_path as input. The input to this tool should be a string, representing the image_path. '  # noqa
#     ),
#     'Text2BoxTool':
#     dict(
#         model='glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
#         description=
#         'useful when you only want to detect or find out given objects in the picture. The input to this tool should be a comma separated string of two, representing the image_path, the text description of the object to be found'  # noqa
#     ),
#     'Text2ImageTool':
#     dict(
#         model='stable_diffusion',
#         description=
#         'useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. The input to this tool should be a string, representing the text used to generate image. '  # noqa
#     ),
#     'OCRTool':
#     dict(
#         model='svtr-small',
#         description=
#         'useful when you want to recognize the text from a photo. receives image_path as inputs. The input to this tool should be a string, representing the image_path. '  # noqa
#     ),
#     'HumanBodyPoseTool':
#     dict(
#         model='human',
#         description=
#         'useful when you want to know the skeleton of a human, or estimate the pose or keypoints of a human. The input to this tool should be a string, representing the image_path. '  # noqa
#     )
# }

DEFAULT_TOOLS = {
    k: v.DEFAULT_TOOLMETA
    for k, v in tools.__dict__.items()
    if inspect.isclass(v) and issubclass(v, BaseTool)
}

NAMES2TOOLS = {
    k: v
    for k, v in tools.__dict__.items()
    if inspect.isclass(v) and issubclass(v, BaseTool)
}

CACHED_TOOLS = defaultdict(dict)


def list_tool():
    return DEFAULT_TOOLS.keys()


def load_tool(tool_name: str,
              *,
              model: Optional[str] = None,
              description: Optional[str] = None,
              input_description: Optional[str] = None,
              output_description: Optional[str] = None,
              device: Optional[str] = 'cpu',
              **kwargs) -> Union[callable, BaseTool]:
    """Load a configurable callable tool for different task.

    Args:
        tool_name (str): tool name for specific task. You can find more
            description about supported tools in `Capability Matrix`_
        model (str, optional): model name defined in OpenMMLab metafile. If it
            is not specified, recommended tool will be loaded according to the
            ``tool``. You can find more description about supported model in
            `Capability Matrix`_. Defaults to None.
        device (str): device to load the model. Defaults to `cpu`.
        **kwargs: key-word arguments to build the specific tools.
            These arguments are related ``tool``. You can find the arguments
            of the specific tool type according to the given tool in the
            `Capability Matrix`_

    Returns:
        Tuple[callable, ToolMeta]: A tuple with callable tool and its meta
        information. The commonly used information by LLM agent like
        "description" can be found in meta.

    Examples:
        >>> from mmlmtools import load_tool
        >>> # load tool with tool name
        >>> tool, meta = load_tool('object detection')
        >>> # load a specific model
        >>> tool, meta = load_tool(
        >>>     'object detection', model='rtmdet_l_8xb32-300e_coco')



    Returns:
        Tuple[callable, ToolMeta]: _description_

    .. _Capability Matrix: TODO
    """
    if tool_name not in NAMES2TOOLS:
        # Using ValueError to show error msg cross lines.
        raise ValueError(f'{tool_name} is not supported now, the available '
                         'tools are:\n' +
                         '\n'.join(map(repr, NAMES2TOOLS.keys())))

    tool_meta = DEFAULT_TOOLS[tool_name]

    if model is None:
        model = tool_meta.get('model', None)

    if description is None:
        description = tool_meta.get('description')

    if input_description is None:
        input_description = tool_meta.get('input_description')

    if output_description is None:
        output_description = tool_meta.get('output_description')

    tool_id = dumps((tool_name, model, model, description, input_description,
                     output_description, device, kwargs))

    tool_type = NAMES2TOOLS[tool_name]

    if tool_id in CACHED_TOOLS[tool_name]:
        return CACHED_TOOLS[tool_name][tool_id]
    else:
        if len(CACHED_TOOLS[tool_name]) != 0:
            _tool_name = f'{tool_name} {len(CACHED_TOOLS[tool_name])+1}'
        else:
            _tool_name = tool_name

        tool_meta = ToolMeta(
            tool_name=_tool_name,
            description=description,
            model=model,
            input_description=input_description,
            output_description=output_description,
        )

        if inspect.isfunction(tool_type):
            # function tool
            tool_obj = tool_type
            tool_obj.toolmeta = tool_meta
        else:
            tool_obj = tool_type(toolmeta=tool_meta, device=device, **kwargs)

        CACHED_TOOLS[tool_name][tool_id] = tool_obj
    return tool_obj


def custom_tool(*,
                tool_name,
                description: Optional[str] = None,
                input_description: Optional[str] = None,
                output_description: Optional[str] = None,
                force=False):
    """Register custom tool.

    Args:
        tool_name (str): The name of tool.
        description (str): The description of the tool.
        force (bool): Whether to overwrite the exists tool with the same name.
            Defaults to False.

    Examples:
        >>> @custom_tool(tool_name="python code executor, description="execute python code")
        >>> def python_code_executor(inputs:)
        >>>     ...
    """  # noqa: E501

    def wrapper(func):
        if tool_name not in DEFAULT_TOOLS:
            DEFAULT_TOOLS[tool_name] = dict(
                tool_name=tool_name,
                description=description,
                input_description=input_description,
                output_description=output_description)
            NAMES2TOOLS[tool_name] = func
        else:
            if not force:
                raise KeyError(
                    'Please do not register tool with duplicated name '
                    f'{tool_name}. If you want to overwrite the old tool, '
                    'please set `force=True`')
            else:
                DEFAULT_TOOLS[tool_name] = dict(
                    tool_name=tool_name,
                    description=description,
                    input_description=input_description,
                    output_description=output_description)
                NAMES2TOOLS[tool_name] = func
        return func

    return wrapper
