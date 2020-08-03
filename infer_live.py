"""
Export transcribed notes from audio file in MIDI / MIREX format.
Because test_evaluate.py does not require the ground truth label,
it does not show the evaluation score.
Usage: python3 infer.py INPUT_FILE_NAME OUTPUT_FILE_NAME [OPTIONS]
    INPUT_FILE_NAME: Input audio file path.
    OUTPUT_FILE_NAME: Output file path.
Options:
    --model-path, -m: Trained model file. If ENSEMBLE=True,
                MODEL_FILE should be directory with model files.
    --ensemble, -e: Choose ensemble method 'mean' and 'vote'.
                    If None, the code will evaluate the single model.
                    (default: None)
    --hop-size, -hs: Choose hop size for melspectrogram.
                     Hop size should be same as the trained model.
                     (default: 512)
    --sample-rate, -sr: Sample rate for input audio.
                        Sample rate should be same as the trained model.
    --onset-threshold, -o: Threshold for onset prediction. (default: 0.5)
    --frame-threshold, -f: Threshold for frame prediction. (default: 0.5)
    --export-midi, -md: The output file format will be MIDI if True.
                        If False, it will export MIREX tsv format.
                        (default: False)
    --device, -d: Choose device for inference.
                  (default: 'cuda' if torch.cuda.is_available() else 'cpu')
Output:
    MIDI file named OUTPUT_FILE_NAME if --export-midi=True.
    tsv file named OUTPUT_FILE_NAME if --expoert-midi=False.
"""
import argparse
import csv
import numpy as np
import torch
import librosa
import math
import os
from onsets_and_frames import MelSpectrogram
from onsets_and_frames.transcriber import load_transcriber
from onsets_and_frames.decoding import extract_notes
from onsets_and_frames.constants import *
from onsets_and_frames.midi import save_midi
from mir_eval.util import midi_to_hz
from pathlib import Path
import matplotlib.pyplot as plt
import time
import matplotlib

UPDATE_FRAMES = 1
NUM_FRAMES = 6

# Transcribe note from input_file in MIREX form / MIDI format with model files.
def note_transcribe(input_file_name, output_file_name, model_path, ensemble,
                    hop_size, sr, onset_threshold, frame_threshold,
                    export_midi, device):
    """
    Transcribe piano notes with deep learning model and write in lines with
    (onset  offset  F0) form or MIDI file.
    """
    audio, _ = librosa.load(input_file_name, sr=sr)    
    model = load_transcriber(model_path).to(device).eval()

    start_time = time.time()
    # Protection code for audio > 1
    if np.max(np.abs(audio)) > 1:
        audio = audio / np.max(np.abs(audio))

    audio_tensor = torch.from_numpy(audio).to(device).unsqueeze(0)
    #print('input_file_name:{}, length:{}sec, sr:{}, size(audio_tensor):{}'.format(input_file_name, size(audio_tensor)/sr, sr, size(audio_tensor)))
    melspec = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH,
                             HOP_LENGTH, mel_fmin=MEL_FMIN,
                             mel_fmax=MEL_FMAX).to(device)
    mel = (melspec(audio_tensor.reshape(-1, audio_tensor.shape[-1])[:, :]).transpose(-1, -2))

    if ensemble is not None:
        model_paths = list(Path(model_path).glob('*.trm'))
        onset_pred = torch.zeros(mel.shape[0], mel.shape[1],
                                 MAX_MIDI - MIN_MIDI + 1).to(device)
        frame_pred = torch.zeros(mel.shape[0], mel.shape[1],
                                 MAX_MIDI - MIN_MIDI + 1).to(device)
        vel_pred = torch.zeros(mel.shape[0], mel.shape[1],
                               MAX_MIDI - MIN_MIDI + 1).to(device)

        for model_path in model_paths:
            model = load_transcriber(model_path).to(device).eval()
            onset_pred_part, _, _, frame_pred_part, vel_pred_part = model(mel)

            if ensemble == 'mean':
                onset_pred += onset_pred_part / len(model_paths)
                frame_pred += frame_pred_part / len(model_paths)
                vel_pred += vel_pred_part / len(model_paths)
            elif ensemble == 'vote':
                # extract_notes does not use offset. -> mean
                onset_pred += ((onset_pred_part > onset_threshold)
                               .type(torch.float))
                frame_pred += ((frame_pred_part > frame_threshold)
                               .type(torch.float))
                vel_pred += vel_pred_part

            del model

    else:
        
        onset_pred, offset_pred, _, frame_pred, vel_pred = model(mel)

    onset_pred = onset_pred.squeeze()
    frame_pred = frame_pred.squeeze()
    vel_pred = vel_pred.squeeze()

    p_est, i_est, v_est = extract_notes(onset_pred, frame_pred,
                                        vel_pred, onset_threshold,
                                        frame_threshold)

    # if export_midi:
    #     scaling = HOP_LENGTH / SAMPLE_RATE

    #     i_est = (i_est * scaling).reshape(-1, 2)
    #     p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    #     save_midi(output_file_name, p_est, i_est, v_est)
    # else:
    #     _write_tsv(p_est, i_est, output_file_name, hop_size, sr)

    new_piano_roll = np.zeros( (frame_pred.shape[1], frame_pred.shape[0]) )
    
    for idx in range(len(p_est)):
        pitch_num = p_est[idx]
        onset_fr  = i_est[idx][0]
        offset_fr = i_est[idx][1]
        velocity  = v_est[idx]
        if velocity != 0:
            new_piano_roll[pitch_num, onset_fr:offset_fr] = 1
    end_time = time.time()    
    filename = output_file_name + '_num_frames' + str(frame_pred.shape[0]) + '.png'
    print('filename:{}, time_elapsed:{}'.format(filename, end_time-start_time))
    matplotlib.image.imsave(filename, new_piano_roll)
    
# Transcribe note from input_file in MIREX form / MIDI format with model files.
def note_transcribe_per_min(input_file_name, output_file_name, model_path, ensemble,
                    hop_size, sr, onset_threshold, frame_threshold,
                    export_midi, device):
    """
    Transcribe piano notes with deep learning model and write in lines with
    (onset  offset  F0) form or MIDI file.
    """
    audio, _ = librosa.load(input_file_name, sr=sr)

    # Protection code for audio > 1
    if np.max(np.abs(audio)) > 1:
        audio = audio / np.max(np.abs(audio))
    
    dur_music_sec = math.floor(len(audio)/sr)
    
    for i in range(round(dur_music_sec)):
        
        print('i:{}, dur_music_sec:{}'.format(i, dur_music_sec))
        print('audio.size():{}, type(audio):{}, sr:{}'.format( audio.shape, type(audio), sr))
        onesec_audio = audio[i*sr:(i+19)*sr]
        print('type(onesec_audio):{}, len(onesec_audio):{}'.format(type(onesec_audio), onesec_audio.shape))
        audio_tensor = torch.from_numpy(onesec_audio).to(device).unsqueeze(0)
        melspec = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH,
                                HOP_LENGTH, mel_fmin=MEL_FMIN,
                                mel_fmax=MEL_FMAX).to(device)
        mel = (melspec(audio_tensor
                    .reshape(-1, audio_tensor.shape[-1])[:, :-1])
            .transpose(-1, -2))

        if ensemble is not None:
            print('mel.shape:{}'.format(mel.shape))
            model_paths = list(Path(model_path).glob('*.trm'))
            onset_pred = torch.zeros(mel.shape[0], mel.shape[1],
                                    MAX_MIDI - MIN_MIDI + 1).to(device)
            frame_pred = torch.zeros(mel.shape[0], mel.shape[1],
                                    MAX_MIDI - MIN_MIDI + 1).to(device)
            vel_pred = torch.zeros(mel.shape[0], mel.shape[1],
                                MAX_MIDI - MIN_MIDI + 1).to(device)

            for model_path in model_paths:
                model = load_transcriber(model_path).to(device).eval()
                onset_pred_part, _, _, frame_pred_part, vel_pred_part = model(mel)

                if ensemble == 'mean':
                    onset_pred += onset_pred_part / len(model_paths)
                    frame_pred += frame_pred_part / len(model_paths)
                    vel_pred += vel_pred_part / len(model_paths)
                elif ensemble == 'vote':
                    # extract_notes does not use offset. -> mean
                    onset_pred += ((onset_pred_part > onset_threshold)
                                .type(torch.float))
                    frame_pred += ((frame_pred_part > frame_threshold)
                                .type(torch.float))
                    vel_pred += vel_pred_part

                del model

        else:
            model = load_transcriber(model_path).to(device).eval()

            onset_pred, offset_pred, _, frame_pred, vel_pred = model(mel)

        onset_pred = onset_pred.squeeze()
        frame_pred = frame_pred.squeeze()
        vel_pred = vel_pred.squeeze()

        print('onset_pred:{}, frame_pred:{}'.format(onset_pred, frame_pred))
        p_est, i_est, v_est = extract_notes(onset_pred, frame_pred,
                                            vel_pred, onset_threshold,
                                            frame_threshold)

        if export_midi:
            scaling = HOP_LENGTH / SAMPLE_RATE
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
            filename, file_extension = os.path.splitext(output_file_name)
            onesec_output_file_name = filename + '_' + str(i) + 'sec' + file_extension
            print('onesec_output_file_name:{}, scaling:{}\n len(i_est):{}, i_est:{}, len(p_est):{}, p_est:{}'.format(onesec_output_file_name, scaling, len(i_est), i_est, len(p_est), p_est))
            save_midi(onesec_output_file_name, p_est, i_est, v_est)
        else:
            _write_tsv(p_est, i_est, output_file_name, hop_size, sr)

# Transcribe note from input_file in MIREX form / MIDI format with model files.
def note_transcribe_per_shiftsize(input_file_name, output_file_name, model_path, ensemble,
                    hop_size, sr, onset_threshold, frame_threshold,
                    export_midi, device):
    """
    Transcribe piano notes with deep learning model and write in lines with
    (onset  offset  F0) form or MIDI file.
    """
    audio, _ = librosa.load(input_file_name, sr=sr)

    # Protection code for audio > 1
    if np.max(np.abs(audio)) > 1:
        audio = audio / np.max(np.abs(audio))
    
    num_chunks = math.floor( (len(audio) - WINDOW_LENGTH)/HOP_LENGTH)
    melspec = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH,
                                HOP_LENGTH, mel_fmin=MEL_FMIN,
                                mel_fmax=MEL_FMAX).to(device)
    model = load_transcriber(model_path).to(device).eval()
    start_time = time.time()
    acc_piano_roll = np.zeros( (88, 4500) )    # 30 sec length 
    for i in range( num_chunks ):       
        start_idx = HOP_LENGTH * i
        end_idx = WINDOW_LENGTH + HOP_LENGTH * (i+NUM_FRAMES)
        chunk_audio = audio[start_idx:end_idx]        
        audio_tensor = torch.from_numpy(chunk_audio).to(device).unsqueeze(0)
        mel_before = time.time()
        mel = (melspec(audio_tensor.reshape(-1, audio_tensor.shape[-1])[:, :]).transpose(-1, -2))
        mel_after = time.time()
        onset_pred, offset_pred, _, frame_pred, vel_pred = model(mel)
        model_after = time.time()
        onset_pred = onset_pred.squeeze()
        frame_pred = frame_pred.squeeze()
        vel_pred = vel_pred.squeeze()
        p_est, i_est, v_est = extract_notes(onset_pred, frame_pred,
                                            vel_pred, onset_threshold,
                                            frame_threshold)
        extract_notes_after = time.time()

        new_piano_roll = np.zeros( (frame_pred.shape[1], frame_pred.shape[0]) )
        print('i:{}, len(chunk_audio):{}, frame_size:{}x{}, mel:{}, model:{}, extract_notes:{}'.format(i, len(chunk_audio), frame_pred.shape[0], frame_pred.shape[1], mel_after- mel_before, model_after - mel_after, extract_notes_after - model_after))
        for idx in range(len(p_est)):
            pitch_num = p_est[idx]
            onset_fr  = i_est[idx][0]
            offset_fr = i_est[idx][1]
            velocity  = v_est[idx]
            if velocity != 0:
                new_piano_roll[pitch_num, onset_fr:offset_fr] = 1
        if NUM_FRAMES == 6:            
            pick_fr_idx = round( frame_pred.shape[0]/2 )-1  
        else:
            pick_fr_idx = round( frame_pred.shape[0]/2 )

        print('pick_fr_idx:{}'.format(pick_fr_idx))       
        acc_piano_roll[:, -UPDATE_FRAMES:] = new_piano_roll[:, pick_fr_idx:pick_fr_idx+1]
        acc_piano_roll[:, :-UPDATE_FRAMES] = acc_piano_roll[:, UPDATE_FRAMES:]
        # if export_midi:
        #     scaling = HOP_LENGTH / SAMPLE_RATE
        #     i_est = (i_est * scaling).reshape(-1, 2)
        #     p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
        #     filename, file_extension = os.path.splitext(output_file_name)
        #     onesec_output_file_name = filename + '_' + str(i) + 'sec' + file_extension            
        #     save_midi(onesec_output_file_name, p_est, i_est, v_est)
        # else:
        #     _write_tsv(p_est, i_est, output_file_name, hop_size, sr)
    end_time = time.time()
    print('total_time:{}'.format(end_time- start_time))
    filename = output_file_name + '_num_frames' + str(NUM_FRAMES) + '.png'
    matplotlib.image.imsave(filename, acc_piano_roll)


def _midi_num_to_f0(midi_num):
    return '{:0.2f}'.format(pow(2, (midi_num - 69 + 21) / 12) * 440)


def _frame_to_time(frame_num, hop_size, sr):
    n_samples = hop_size * frame_num
    secs = n_samples / sr
    return '{:0.2f}'.format(secs)


# MIREX format
def _write_tsv(pitch, interval, output_file_name, hop_size, sr):
    with open(output_file_name, 'wt') as output_file:
        tsv_writer = csv.writer(output_file, delimiter='\t')
        for i in range(len(pitch)):
            # Each row contains onset, offset, and the F0
            tsv_writer.writerow([_frame_to_time(interval[i][0], hop_size, sr),
                                 _frame_to_time(interval[i][1], hop_size, sr),
                                 _midi_num_to_f0(pitch[i])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_name',
                        default='test_resource/test_audio.wav', type=str)
    parser.add_argument('output_file_name', default='output.f0', type=str)

    parser.add_argument('--model-path', '-m', type=str,
                        default='test_resource/model.pt')
    parser.add_argument('--ensemble', '-e', type=str, default=None)
    parser.add_argument('--hop-size', '-hs',
                        default=512, type=int)
    parser.add_argument('--sample-rate', '-sr', default=16000, type=int)
    parser.add_argument('--onset-threshold', '-o', default=0.5, type=float)
    parser.add_argument('--frame-threshold', '-f', default=0.5, type=float)
    parser.add_argument('--export-midi', '-md', default=False, type=bool)
    parser.add_argument('--device', '-d', default='cuda'
                        if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    with torch.no_grad():
        #note_transcribe(args.input_file_name, args.output_file_name,
        note_transcribe_per_shiftsize(args.input_file_name, args.output_file_name,        
                        args.model_path, args.ensemble, args.hop_size,
                        args.sample_rate, args.onset_threshold,
                        args.frame_threshold, args.export_midi, args.device)