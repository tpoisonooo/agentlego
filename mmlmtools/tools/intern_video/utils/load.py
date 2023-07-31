# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

try:
    from decord import VideoReader, cpu
    has_decord = True
except ImportError:
    has_decord = False


class LoadVideo:

    def __init__(self):
        if not has_decord:
            raise ImportError('Please install decord to enable video loading.')
        self.video_path = None
        self.data = None

    def __call__(self, video_path):
        if self.video_path == video_path:
            return self.data
        self.data = self.load_original_video_decord(video_path)
        self.video_path = video_path
        return self.data

    def load_original_video_decord(self,
                                   sample,
                                   sample_rate_scale=1,
                                   clip_len=8,
                                   frame_sample_rate=2,
                                   num_segment=1):
        fname = sample
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        # handle temporal segments
        # converted_len = int(clip_len * frame_sample_rate)
        seg_len = len(vr) // num_segment
        duration = max(len(vr) // vr.get_avg_fps(), 8)

        all_index = []
        for i in range(num_segment):
            index = np.linspace(0, seg_len, num=int(duration))
            index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer
