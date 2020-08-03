import sys
from functools import reduce

import torch
import numpy as np
from PIL import Image
from torch.nn.modules.module import _addindent
from .mel import melspectrogram


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def mixup(batch, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        audio_label = batch['audio']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        batch['mel'] = mel

        return batch, 0

    audio_label = batch['audio']
    mel_label = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_label = batch['onset']
    offset_label = batch['offset']
    frame_label = batch['frame']
    velocity_label = batch['velocity']

    batch_size = mel_label.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_mel = lam * mel_label + (1 - lam) * mel_label[index, :]
    mixed_onset = lam * onset_label + (1 - lam) * onset_label[index, :]
    mixed_offset = lam * offset_label + (1 - lam) * offset_label[index, :]
    mixed_frame = lam * frame_label + (1 - lam) * frame_label[index, :]
    mixed_vel = lam * velocity_label + (1 - lam) * velocity_label[index, :]

    m_batch = dict(path=batch['path'], audio=batch['audio'], mel=mixed_mel,
                   onset=mixed_onset, offset=mixed_offset, frame=mixed_frame,
                   velocity=mixed_vel)

    return m_batch, lam


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    onsets = (1 - (onsets.t() > onset_threshold).type(torch.int)).cpu()
    frames = (1 - (frames.t() > frame_threshold).type(torch.int)).cpu()
    both = (1 - (1 - onsets) * (1 - frames))
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
