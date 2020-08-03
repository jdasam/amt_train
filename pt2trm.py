import torch
import argparse
from onsets_and_frames.constants import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str,
                        help='Previous model file path to fix.')
    parser.add_argument('save_name', type=str,
                        help='Path for new model file.')
    parser.add_argument('input_features', type=int, nargs='?', default=N_MELS,
                        help='Input features of model.')
    parser.add_argument('output_features', type=int, nargs='?',
                        default=MAX_MIDI - MIN_MIDI + 1,
                        help='Output features of model.')

    args = parser.parse_args()

    model = torch.load(args.model_file)

    model.input_features = args.input_features
    model.output_features = args.output_features

    try:
        model_size_conv = model.onset_stack[0].fc[0].out_features
    except TypeError:
        model_size_conv = model.onset_stack.module[0].fc[0].out_features
    try:
        model_size_lstm = model.onset_stack[-2].in_features
    except TypeError:
        model_size_lstm = model.onset_stack.module[-2].in_features

    model.model_complexity_conv = model_size_conv // 16
    model.model_complexity_lstm = model_size_lstm // 16

    model.save(args.save_name)
