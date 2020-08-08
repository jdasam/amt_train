import os
import numpy as np

from torch.nn import BCELoss, DataParallel, CrossEntropyLoss
from onsets_and_frames import *
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from evaluate import evaluate
from metalearner.common.config import experiment, worker
from metalearner.api import scalars
# from meta_reporter import MetaReporter
from config import ex
from sacred.commands import print_config


g_max_note_f1_maps = 0
g_max_note_overlap_maps = 0
g_max_note_onset_offset_f1_maps = 0
g_max_note_f1_maestro = 0
g_max_note_overlap_maestro = 0
g_max_note_onset_offset_f1_maestro = 0
inside_idx = 0
sample_note_f1 = [10, 100, 20]
sample_note_overlap = [20, 200, 30]
sample_note_onoffset_f1 = [30, 300, 40]


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval,
          batch_size, sequence_length, model_complexity_conv,
          model_complexity_lstm, use_gpu, dataset_list,
          valid_dataset_list, learning_rate, learning_rate_decay_steps,
          learning_rate_decay_rate, leave_one_out, clip_gradient_norm,
          validation_length, validation_interval, d_learning_rate,
          pix2pix_weight, mixup_strength, worker_id):
    print_config(ex.current_run)

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = (str(use_gpu).strip('[]')
    #                                                   .replace(' ', ''))
    torch.cuda.set_device(use_gpu[0])

    use_dp = False if len(use_gpu) < 2 else True

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']
    test_groups = ['test']

    # MetaReporter().initialize(worker_id)

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011',
                     '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    # For train dataset 채우기 
    if len(dataset_list) == 2:
        maestro_train_dataset = MAESTRO(groups=train_groups,
                                        sequence_length=sequence_length)
        maps_train_dataset = MAPS(groups=train_groups,
                                  sequence_length=sequence_length)
        dataset = ConcatDataset((maestro_train_dataset, maps_train_dataset))
    elif dataset_list[0] == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
    elif dataset_list[0] == 'MAPS':
        dataset = MAPS(groups=train_groups, sequence_length=sequence_length)
    else:
        raise ValueError('Invalid dataset name.')

    # For valid dataset 채우기 
    maestro_valid_dataset = MAESTRO(groups=test_groups,
                                    sequence_length=sequence_length)
    # maps_valid_dataset = MAPS(groups=test_groups, sequence_length=sequence_length)
    #maps_maestro_valid_dataset = ConcatDataset((maestro_valid_dataset,
    #                                            maps_valid_dataset))
    # Build batch for loader
    print('The Dataset size is train: {}, valid: {}'.format(len(dataset), len(maestro_valid_dataset)))
    loader = DataLoader(dataset, batch_size, shuffle=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, 
                                model_complexity_conv,
                                model_complexity_lstm).to(device)
        if use_dp:
            model = DataParallel(model, use_gpu)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
        
    else:
        model_path = os.path.join(logdir,
                                  'model-{}.pt'.format(resume_iteration))
        model = load_transcriber(model_path, use_dp).to(device)
        if use_dp:
            model = DataParallel(model, use_gpu)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path .join(logdir,'last-optimizer-state.pt')))


    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps,
                       gamma=learning_rate_decay_rate)


    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    criterion = CrossEntropyLoss()
    for i, batch in zip(loop, cycle(loader)):
        # batch, _ = utils.mixup(batch, mixup_strength, device)
        if use_dp:
            audio_label = batch['audio']
            if 'mel' in batch.keys():
                mel_label = batch['mel']
            else:
                batch['mel'] = model.module.melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1])\
                            .transpose(-1, -2)
                mel_label = batch['mel']
            state_label = batch['label']
            predictions = model(mel_label, state_label) # [mel_label: ? , state_label: ?]
            pred = predictions.permute(0,3,1,2)
            target = state_label.type(torch.LongTensor).to(predictions.device)
            # pred = label_pred.view(-1, 5)          # [56320, 5] , label_pred = [1, 640, 88, 5], N x C number of classes
            # target = state_label.type(torch.LongTensor).to(label_pred.device).view(-1) # [56320], 0 <= target <= C-1 values 
            loss = criterion(pred, target)
        else:
            predictions, loss = model.run_on_batch(batch)
        # loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)   #Clips gradient norm of an iterable of parameters.
        
        writer.add_scalar('train_loss', loss.item(), global_step=i)
        # for key, value in {'loss': loss, **losses}.items():
        #     writer.add_scalar(key, value.item(), global_step=i)

        global g_max_note_f1_maps, g_max_note_overlap_maps
        global g_max_note_onset_offset_f1_maps
        global g_max_note_f1_maestro, g_max_note_overlap_maestro
        global g_max_note_onset_offset_f1_maestro
        global sample_note_f1, sample_note_overlap_f1
        global sample_note_onoffset_f1, inside_idx

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                note_f1 = 0
                note_recall = 0
                note_onset_offset_f1 = 0
                note_precision = 0
                for key, value in evaluate(model, maestro_valid_dataset).items():
                # for key, value in evaluate(model, maps_valid_dataset).items():
                    writer.add_scalar('maestro_test/' + key.replace(' ', '_'),
                                       np.mean(value), global_step=i)
                    if (key == 'metric/note/f1'):
                        note_f1 = np.mean(value)
                    if (key == 'metric/note/recall'):
                        note_recall = np.mean(value)
                    if (key == 'metric/note/precision'):
                        note_precision = np.mean(value)
                    if (key == 'metric/note-with-offsets/f1'):
                        note_onset_offset_f1 = np.mean(value)
                results = {'note_f1': note_f1, 
                           'note-with-offsets': note_onset_offset_f1, 
                           'note_recall': note_recall,
                           'note_precision': note_precision}
                response = scalars.send_valid_result(worker.id, i//len(dataset), i, results)
            model.train()
            

        if i % checkpoint_interval == 0:
            if use_dp:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_complexity_conv': model_complexity_conv,
                'model_complexity_lstm': model_complexity_lstm,
            }
            , os.path.join(logdir, 'model-{}.pt'.format(i)))
            torch.save(optimizer.state_dict(),
                       os.path.join(logdir, 'last-optimizer-state.pt'))
        # For MetaReport, send the evaluation metric values once at the end of training.
        # if i == iterations:
        #     MetaReporter().send_iteration(i, g_max_note_f1_maps,
        #                                   g_max_note_overlap_maps,
        #                                   g_max_note_onset_offset_f1_maps,
        #                                   g_max_note_f1_maestro,
        #                                   g_max_note_overlap_maestro,
        #                                   g_max_note_onset_offset_f1_maestro)
        #     print(
        #         'kch debug max_iter:{}, g_max_note_f1_maps:{}, g_max_note_overlap_maps:{}, g_max_note_onset_offset_f1_maps:{}'.format(
        #             iterations, g_max_note_f1_maps, g_max_note_overlap_maps, g_max_note_onset_offset_f1_maps))
        #     print(
        #         'kch debug max_iter:{}, g_max_note_f1_maestro:{}, g_max_note_overlap_maestro:{}, g_max_note_onset_offset_f1_maestro:{}'.format(
        #             iterations, g_max_note_f1_maestro, g_max_note_overlap_maestro,
        #             g_max_note_onset_offset_f1_maestro))