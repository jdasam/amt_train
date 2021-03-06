"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/
        onsets_and_frames/onsets_frames_transcription/onsets_and_frames.py

Removed DataParallel from transcriber.py to run the model on the CPU.
Detailed information about the class is in transcriber.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16,
                      (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8,
                      (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4),
                      output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFramesNoDP(nn.Module):
    def __init__(self, input_features, output_features,
                 model_complexity_conv=48, model_complexity_lstm=48):
        super().__init__()

        def _sequence_model(input_size, output_size):
            return BiLSTM(input_size, output_size // 2)

        model_size_conv = model_complexity_conv * 16
        model_size_lstm = model_complexity_lstm * 16

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size_conv),
            _sequence_model(model_size_conv, model_size_lstm),
            nn.Linear(model_size_lstm, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size_conv),
            _sequence_model(model_size_conv, model_size_lstm),
            nn.Linear(model_size_lstm, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size_conv),
            nn.Linear(model_size_conv, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            _sequence_model(output_features * 3, model_size_lstm),
            nn.Linear(model_size_lstm, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size_conv),
            nn.Linear(model_size_conv, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(),
                                   offset_pred.detach(), activation_pred],
                                  dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return (onset_pred, offset_pred, activation_pred,
                frame_pred, velocity_pred)

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        if 'mel' in batch.keys():
            mel_label = batch['mel']
        else:
            batch['mel'] = melspectrogram(audio_label
                                          .reshape(-1,
                                                   audio_label
                                                   .shape[-1])[:, :-1])\
                           .transpose(-1, -2)
        mel_label = batch['mel']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel_label)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'],
                                                 onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'],
                                                  offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'],
                                                 frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'],
                                                velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label *
                    (velocity_label - velocity_pred) ** 2).sum() / denominator

    def save(self, filename):
        try:
            onset_state_dict = self.onset_stack.module.state_dict()
            offset_state_dict = self.offset_stack.module.state_dict()
            frame_state_dict = self.frame_stack.module.state_dict()
            combined_state_dict = self.combined_stack.module.state_dict()
            velocity_state_dict = self.velocity_stack.module.state_dict()
        except AttributeError:
            onset_state_dict = self.onset_stack.state_dict()
            offset_state_dict = self.offset_stack.state_dict()
            frame_state_dict = self.frame_stack.state_dict()
            combined_state_dict = self.combined_stack.state_dict()
            velocity_state_dict = self.velocity_stack.state_dict()

        # Save state_dict in cpu
        state_dict_list = [onset_state_dict, offset_state_dict,
                           frame_state_dict, combined_state_dict,
                           velocity_state_dict]
        for state_dict in state_dict_list:
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

        parameters = dict(onset=onset_state_dict, offset=offset_state_dict,
                          frame=frame_state_dict, combined=combined_state_dict,
                          velocity=velocity_state_dict,
                          input_features=self.input_features,
                          output_features=self.output_features,
                          model_complexity_conv=self.model_complexity_conv,
                          model_complexity_lstm=self.model_complexity_lstm)

        torch.save(parameters, filename)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator_network = nn.Sequential(
            # layer 0
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 1
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(),
            nn.LeakyReLU(negative_slope=0.2),
            # layer 4
            nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, score):
        x = self.discriminator_network(score)
        x = torch.mean(x, dim=(1, 2, 3))
        x = self.sigmoid(x)
        return x
