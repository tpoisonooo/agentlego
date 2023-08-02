from unittest import skipIf

from mmengine import is_installed

from mmlmtools import load_tool
from mmlmtools.testing import ToolTestCase
from mmlmtools.tools.parsers import NaiveParser


@skipIf(not is_installed('transformers'), reason='requires `transformers`')
class TestVideoCaptionTool(ToolTestCase):

    def test_call(self):
        tool = load_tool(
            'VideoCaptionTool', parser=NaiveParser(), device='cuda:0')
        print(tool.name)
        print(tool.inputs)
        print(tool.outputs)
        print(tool.description)

        vid_path = 'tests/data/video/football.mp4'
        inputs = (vid_path, )
        outputs = tool(inputs)
        print(outputs)
