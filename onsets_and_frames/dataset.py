import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .paths import *
from .midi import parse_midi

import matplotlib.pyplot as plt 
import matplotlib as mpl


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None,
                 seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []
        print('Loading %d group%s of %s at %s'
              % (len(groups), 's'[:len(groups)-1],
                 self.__class__.__name__, path))
        for group in groups:
            for input_files in tqdm(self.files(group),
                                    desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])  # 541741

            # audio_length v.s. sequence_length protection code (19/09/12 by KCH)
            if (audio_length - self.sequence_length) <= 0:
                step_begin = 0
            else:
                step_begin = (self.random.randint(audio_length -
                                                  self.sequence_length) //
                              HOP_LENGTH)
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = (data['label'][step_begin:step_end, :]
                               .to(self.device))
            # result['velocity'] = (data['velocity'][step_begin:step_end, :]
            #                       .to(self.device))
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            # result['velocity'] = data['velocity'].to(self.device).float()

        result['audio']  = result['audio'].float().div_(32768.0)
        # result['onset']  = (result['label'] == 3).float()
        # result['offset'] = (result['label'] == 1).float()
        # result['sustain'] = (result['label'] == 2).float()
        # result['reonset'] = (result['label'] == 4).float()                
        # result['off']     = (result['label'] == 0).float()
        # result['frame']    = (result['label'] > 1).float()
        # result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename)
           for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            ramp: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the number of frames
                after the corresponding onset

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values
                at the frame locations
        """
        saved_data_path = (audio_path.replace('.flac', '.pt2')
                                     .replace('.wav', '.pt2'))
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        # Sequence length protection code (19/09/12 by KCH)
        if ((self.sequence_length is not None) and
           (len(audio) < self.sequence_length)):
            audio = np.pad(audio, (0, self.sequence_length - len(audio)), 'constant', constant_values=(0,0))

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        
        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        # # 비교용 원래 reonset (4) 이 없을 때
        orig_label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        orig_velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        
        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
            
            f = int(note) - MIN_MIDI            
            
            orig_label[left:onset_right, f] = 3
            orig_label[onset_right:frame_right, f] = 2
            orig_label[frame_right:offset_right, f] = 1
            orig_velocity[left:frame_right, f] = vel            

        note_idx = 0
        
        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
            
            f = int(note) - MIN_MIDI
            
            note_idx = note_idx + 1            
            if label[left:onset_right, f] == 4: # 이미 reonset (4) 으로 check 되어 있으면 다음 note로 continue
                continue

            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel
            
            # midi.shape = events x 4 (onset, offset, note, vel) 이고 비교 노트는 본 노트 다음 부터 고려 하여
            # 만약 동일 pitch의 비교 노트 onset (==left)이 본 노트의 onset (==left)와 offset (==offset_right) 사이에 있으면 reonset (4)으로 설정            
            for cmp_onset, cmp_offset, cmp_note, cmp_vel in midi[note_idx:,:]:
                cmp_left = int(round(cmp_onset * SAMPLE_RATE / HOP_LENGTH))
                cmp_onset_right = min(n_steps, cmp_left + HOPS_IN_ONSET)
                cmp_frame_right = int(round(cmp_offset * SAMPLE_RATE / HOP_LENGTH))
                cmp_frame_right = min(n_steps, cmp_frame_right)
                cmp_offset_right = min(n_steps, cmp_frame_right + HOPS_IN_OFFSET)
            
                cmp_f = int(cmp_note) - MIN_MIDI               

                if (cmp_f == f) and (cmp_left > left) and (cmp_left < offset_right):
                    label[cmp_left:cmp_onset_right, f] = 4                    
                    label[cmp_onset_right:cmp_frame_right, f] = 2
                    label[cmp_frame_right:cmp_offset_right, f] = 1
                    velocity[cmp_left:cmp_frame_right, f] = cmp_vel
                    # repeat note vel이 본노트 vel 보다 크고 본노트와 repeat note의 간격이 96msec (=3frames 이상인경우) 추가적으로 reonset 앞에 offset을 주어 끊고 가기  
                    if (velocity[cmp_left, f] > velocity[left, f]) and (cmp_left - left >= 3):                        
                        label[cmp_left-1:cmp_left, f] = 1
                    
        # fig = plt.figure()
        # ax1 = fig.add_subplot(2,1,1)
        # tmp_orig_label = orig_label.transpose(1,0)
        # im1 = ax1.imshow( tmp_orig_label[41:51,:300], cmap='hot', origin='lower', aspect='auto')
        # fig.colorbar(im1, ax=ax1)
        
        # ax2 = fig.add_subplot(2,1,2)
        # tmp_label = label.transpose(1,0)
        # im2 = ax2.imshow( tmp_label[41:51,:300], cmap='hot', origin='lower', aspect='auto')        
        # fig.colorbar(im2, ax=ax2)

        # fig.savefig('re_cmp_figure.png')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(2,1,1)
        # tmp_orig_label = orig_label.transpose(1,0)
        # im1 = ax1.imshow( tmp_orig_label[41:51,1300:1600], cmap='hot', origin='lower', aspect='auto')
        # fig.colorbar(im1, ax=ax1)       
        
        # ax2 = fig.add_subplot(2,1,2)
        # tmp_label = label.transpose(1,0)
        # im2 = ax2.imshow( tmp_label[41:51,1300:1600], cmap='hot', origin='lower', aspect='auto')        
        # fig.colorbar(im2, ax=ax2)

        # fig.savefig('re_cmp_figure2.png')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(2,1,1)
        # tmp_orig_label = orig_label.transpose(1,0)
        # im1 = ax1.imshow( tmp_orig_label[41:51,3300:3600], cmap='hot', origin='lower', aspect='auto')
        # fig.colorbar(im1, ax=ax1)
        
        # ax2 = fig.add_subplot(2,1,2)
        # tmp_label = label.transpose(1,0)
        # im2 = ax2.imshow( tmp_label[41:51,3300:3600], cmap='hot', origin='lower', aspect='auto')        
        # fig.colorbar(im2, ax=ax2)

        # fig.savefig('re_cmp_figure3.png')

        data = dict(path=audio_path, audio=audio,
                    label=label, velocity=velocity)

        saved_data_path = (audio_path.replace('.flac', '.pt2'))
        
        print('firstly save file@{}'.format(saved_data_path))
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path=MAESTRO_PATH, groups=None,
                 sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'],
                         sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        print('group is', group)
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError('Group ' + group + ' is empty')
        else:
            metadata = json.load(open(os.path.join(self.path,
                                                   'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path,
                                          (row['audio_filename']
                                           .replace('.wav', '.flac'))),
                             os.path.join(self.path,
                                          row['midi_filename']))
                            for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio)
                      else audio.replace('.flac', '.wav'), midi)
                     for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = (midi_path.replace('.midi', '.tsv')
                                     .replace('.mid', '.tsv'))
            if not os.path.exists(tsv_filename):            
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f',
                           delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path=MAPS_PATH, groups=None,
                 sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        # self.subgroup = {'train': ['AkPnBcht2'],
        #                  'validation': ['ENSTDkAm2'],
        #                  'test': ['ENSTDkAm2']}
        self.subgroup = {'train': ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD',
                                  'AkPnStgb', 'SptkBGAm', 'SptkBGCl',
                                  'StbgTGd2'],
                        'validation': ['ENSTDkAm', 'ENSTDkCl'],
                        'test': ['ENSTDkAm', 'ENSTDkCl']}
        super().__init__(path, groups if groups is not None else ['test'],
                         sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']
        # return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm',
        # 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = []
        for subgroup in self.subgroup[group]:
            flacs.extend(glob(os.path.join(self.path,
                                           'flac', '*_%s.flac' % subgroup)))
        tsvs = [f.replace('/flac/',
                          '/tsv/new_matched/').replace('.flac',
                                                       '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))
