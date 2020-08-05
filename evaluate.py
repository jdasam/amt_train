import argparse
import os
import sys
from collections import defaultdict
import numpy as np

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap \
                                as evaluate_notes
from mir_eval.transcription import match_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap \
                                         as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames.mel import melspectrogram

from torch.utils.data import ConcatDataset
from onsets_and_frames import *

import matplotlib.pyplot as plt
from pathlib import Path


eps = sys.float_info.epsilon


def _predict_each_label(label, model):
    """
    Infer each audio label with model.

    Parameters
    ----------
    label: Label from dataset. Contains audio and ground truth label.
    model: Model information. Depends on ensemble and pre_pred.
    ensemble: If ensemble is None, model will be model file.
              If ensemble is not None, model will be the list of models.
    device: Device for models and data.
    pre_pred: If pre_pred is True, pre-predicted result is used for inference.

    Returns
    -------
    pred: Prediction result in dict.
    """

    # pred = model(label['mel'].unsqueeze(0))
    # pred = model.run_on_batch(label, evaluation=True)
    if 'mel' in label.keys():
        mel_label = label['mel']
    else:
        try:
            label['mel'] = model.melspectrogram(label['audio'].reshape(-1, label['audio'].shape[-1])[:, :-1])\
                        .transpose(-1, -2)
        except:
            label['mel'] = model.module.melspectrogram(label['audio'].reshape(-1, label['audio'].shape[-1])[:, :-1])\
                        .transpose(-1, -2)
        mel_label = label['mel']
    pred = model(mel_label)
    pred[pred==4]=3
    return {
        'onset': (pred==3).type(torch.int),
        'frame': (pred>1).type(torch.int),
        'offset': (pred==1).type(torch.int)
    }



def _export_prediction(pred, label, save_path, model_file, dataset_name):
    """
    Export predicted result to files in save_path.

    Parameters
    ----------
    pred: Prediction result.
    label: Ground truth label.
    save_path: Save path for prediction results.
    model_file: File name for model. For creating directory name.
    dataset_name: Dataset name. For creating directory name.
    """
    if not Path(save_path + '/' + Path(model_file).stem +
                '_' + dataset_name).exists():
        os.makedirs(save_path + '/' + Path(model_file).stem + '_' +
                    dataset_name)

    pred['onset'] = pred['onset'].to('cpu')
    pred['offset'] = pred['offset'].to('cpu')
    pred['frame'] = pred['frame'].to('cpu')
    pred['velocity'] = pred['velocity'].to('cpu')

    torch.save(pred, save_path + '/' + Path(model_file).stem + '_' +
               dataset_name + '/' + (Path(label['path']).stem + '_pred.pt'))


def _export_midi(label, pred, save_path, onset_threshold, frame_threshold):
    """
    Export MIDI file and image files in save_path.
    For image export, it calls _export_image function.

    Parameters
    ----------
    pred: Prediction result.
    label: Ground truth label.
    save_path: Save path for prediction results.
    onset_threshold: Threshold for onset prediction.
    frame_threshold: Threshold for frame prediction.
    """
    os.makedirs(save_path, exist_ok=True)
    midi_path = str(Path(save_path) / Path(label['path']).stem)

    label_image = (label['onset'] * 2 + label['frame'] -
                   label['onset'] * label['frame'])
    pred_image = (pred['onset'] * 2 + pred['frame'] -
                  pred['onset'] * pred['frame'])
    _export_image(label_image, pred_image, midi_path + '.png')

    p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'],
                                        pred['velocity'], onset_threshold,
                                        frame_threshold)

    i_est, p_est, _, _ = _rescale_notes(i_est, p_est, pred['frame'].shape)

    save_midi(midi_path + '_pred.mid', p_est, i_est, v_est)


def _export_image(label_image, pred_image, save_path):
    """
    Export MIDI file and image files in save_path.

    Parameters
    ----------
    pred_image: Prediction result in 2D array.
    label_image: Ground truth label in 2D array.
    save_path: Save path for prediction results.
    """
    plt.subplot(211)
    plt.title('label')
    plt.imshow(label_image.cpu().T, origin='lower', aspect='auto')
    plt.subplot(212)
    plt.title('pred')
    plt.imshow(pred_image.cpu().T, origin='lower', aspect='auto')

    plt.savefig(save_path)


def _rescale_notes(intervals, pitches, frames):
    """
    Rescale notes to from raw prediction results to make MIDI.

    Parameters
    ----------
    intervals: Intervals from extract_notes.
    pitches: Pitches from extract_notes.
    frames: Frame prediciton result.

    Returns
    -------
    re_intervals: Rescaled intervals.
    re_pitches: Rescaled pitches.
    re_times: Rescaled times.
    re_freqs: Rescaled frequencies.
    """
    scaling = HOP_LENGTH / SAMPLE_RATE

    re_intervals = (intervals * scaling).reshape(-1, 2)
    re_pitches = np.array([midi_to_hz(MIN_MIDI + midi) for midi in pitches])

    times, freqs = notes_to_frames(pitches, intervals, frames)
    re_times = times.astype(np.float64) * scaling
    re_freqs = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freq])
                for freq in freqs]

    return re_intervals, re_pitches, re_times, re_freqs


def _evaluate_metrics(metrics, pred, label, onset_threshold, frame_threshold):
    """
    Evaluate diverse metrics.

    Parameters
    ----------
    metrics: Dict to store metric for each song.
    label: Ground truth label.
    pred: Prediction result.
    onset_threshold: Threshold for onset prediction.
    frame_threshold: Threshold for frame prediction.
    """
    for key, value in pred.items():
        if key == 'path':
            continue
        value.squeeze_(0).relu_()
    label_onset = (label == 3).float()
    label_frame  = (label > 1).float()
    p_ref, i_ref = extract_notes(label_onset, label_frame)
    p_est, i_est = extract_notes(pred['onset'], pred['frame'],onset_threshold,
                                        frame_threshold)

    i_ref, p_ref, t_ref, f_ref = _rescale_notes(i_ref, p_ref,
                                                label_frame.shape)
    i_est, p_est, t_est, f_est = _rescale_notes(i_est, p_est,
                                                pred['frame'].shape)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est,
                                offset_ratio=None)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'].append(p)
    metrics['metric/note-with-offsets/recall'].append(r)
    metrics['metric/note-with-offsets/f1'].append(f)
    metrics['metric/note-with-offsets/overlap'].append(o)

    # if len(match_notes(i_ref, p_ref,
    #                    i_est, p_est, offset_ratio=None)) == 0:
    #     p, r, f, o = 0.0, 0.0, 0.0, 0.0
    # else:
    #     p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref,
    #                                               i_est, p_est, v_est,
    #                                               offset_ratio=None,
    #                                               velocity_tolerance=0.1)
    # metrics['metric/note-with-velocity/precision'].append(p)
    # metrics['metric/note-with-velocity/recall'].append(r)
    # metrics['metric/note-with-velocity/f1'].append(f)
    # metrics['metric/note-with-velocity/overlap'].append(o)

    # if len(match_notes(i_ref, p_ref, i_est, p_est)) == 0:
    #     p, r, f, o = 0.0, 0.0, 0.0, 0.0
    # else:
    #     p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref,
    #                                               i_est, p_est, v_est,
    #                                               velocity_tolerance=0.1)
    # metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
    # metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
    # metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
    # metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] +
                                            eps, frame_metrics['Recall'] +
                                            eps]) - eps)

    for key, loss in frame_metrics.items():
        metrics['metric/frame/' + (key.lower()
                                   .replace(' ', '_'))].append(loss)


def evaluate(model, data, dataset_name=None, onset_threshold=0.5,
             frame_threshold=0.5, ensemble=None, device='cpu', save_path=None,
             pre_pred=False, export_pred=False, export_midi=False):
    metrics = defaultdict(list)
    i = 0
    for label in data:
        i = i + 1
        audio_len = len(label['audio'])
        file_name = label['path']
        # print('{} {} - {}'.format(i, audio_len, file_name))
        pred = _predict_each_label(label, model)
        
        # pred = pred.cpu().numpy()
        # target = label['label'].cpu().numpy()
        # num_sustain_pred = np.sum(pred==2)   
        # num_attack_pred = np.sum(pred==3)
        # num_sustain_target = np.sum(target==2)
        # num_attack_target = np.sum(target==3)  

        # num_correct_sustain = np.sum(np.logical_and(target==2, pred==2))   
        # num_correct_attack = np.sum(np.logical_and(target==3, pred==3))   

        # sustain_precision = num_correct_sustain / (num_sustain_pred + 1)    
        # sustain_recall = num_correct_sustain / num_sustain_target  
        # attack_precision = num_correct_attack / (num_attack_pred + 1)
        # attack_recall = num_correct_attack / num_attack_target

        # sustain_f1 = 2*sustain_precision *sustain_recall / (sustain_precision + sustain_recall + 0.001)
        # attack_f1 = 2*attack_precision *attack_recall / (attack_precision + attack_recall + 0.001)
        
        # metrics['metric/onset/f1'].append(attack_f1)
        # metrics['metric/onset/precision'].append(attack_precision)
        # metrics['metric/onset/recall'].append(attack_recall)
        # metrics['metric/frame/f1'].append(sustain_f1)
        # metrics['metric/frame/precision'].append(sustain_precision)
        # metrics['metric/frame/recall'].append(sustain_recall)
        
        #save in metrics

        # if export_pred:
        #     _export_prediction(pred, label, save_path,
        #                        model_file, dataset_name)

        _evaluate_metrics(metrics, pred, label['label'],
                          onset_threshold, frame_threshold)

        # if export_midi:
        #     _export_midi(label, pred, save_path,
        #                  onset_threshold, frame_threshold)

    return metrics


def evaluate_model(model_file, dataset_name, dataset_group, sequence_length,
                   onset_threshold, frame_threshold, ensemble, device,
                   use_dp, save_path, pre_pred, export_pred, export_midi):

    # Load dataset
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    if dataset_name == 'MAPS':
        dataset = dataset_module.MAPS(**kwargs)
    elif dataset_name == 'MAESTRO':
        dataset = dataset_module.MAESTRO(**kwargs)
    elif dataset_name == 'ALL':
        dataset_maps = dataset_module.MAPS(**kwargs)
        dataset_maestro = dataset_module.MAESTRO(**kwargs)
        dataset = ConcatDataset((dataset_maps, dataset_maestro))
    else:
        raise Exception('Invalid dataset name.')

    # If pre_pred, load pre-predicted *.pt files instead of load model.
    if pre_pred:
        if ensemble is not None:
            model = list(Path(model_file).glob('*/'))
        else:
            model = model_file
    else:
        if ensemble is not None:
            model = []
            model_names = Path(model_file).glob('*.trm')
            for model_name in model_names:
                model.append(load_transcriber(model_name, use_dp)
                             .to(device).eval())
        else:
            model = load_transcriber(model_file, use_dp).eval().to(device)
            summary(model)

    # Evaluate and get metrics
    metrics = evaluate(model, tqdm(dataset), dataset_name, onset_threshold,
                       frame_threshold, ensemble, device, save_path, pre_pred,
                       export_pred, export_midi)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print('{:>32} {:25}: {:.3f} ± {:.3f}'.format(category, name,
                                                         np.mean(values),
                                                         np.std(values)))

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str,
                        help='Path of model file or pre-predicted *.trm files')
    parser.add_argument('dataset_name', nargs='?', default='MAPS',
                        help='Name of dataset for evaluation.')
    parser.add_argument('dataset_group', nargs='?', default=None,
                        help='Name of dataset group for evaluation.')
    parser.add_argument('--sequence-length', '-sl', default=None, type=int,
                        help='Length of sequences of each audio file.')
    parser.add_argument('--onset-threshold', '-o', default=0.5, type=float,
                        help='Threshold for onset prediction.')
    parser.add_argument('--frame-threshold', '-f', default=0.5, type=float,
                        help='Threshold for frame prediction.')
    parser.add_argument('--ensemble', '-e', default=None, type=str,
                        help='Ensemble type. mean and vote are supported.')
    parser.add_argument('--device', '-d', default='cuda'
                        if torch.cuda.is_available() else 'cpu',
                        help='Device for calculation. ex. "cuda:0" for gpu 0')
    parser.add_argument('--use-dp', '-dp', action='store_true',
                        help='Use data parallel or not.')
    parser.add_argument('--save-path', '-s', default=None,
                        help='Save path for the result if you want to export.')
    parser.add_argument('--pre-pred', '-pp', action='store_true',
                        help='Use pre-predicted result for evaluation.')
    parser.add_argument('--export-pred', '-p', action='store_true',
                        help='Export predicted result files in save path.')
    parser.add_argument('--export-midi', '-m', action='store_true',
                        help='Export MIDI and images of prediction result.')

    with torch.no_grad():
        evaluate_model(**vars(parser.parse_args()))
