from onsets_and_frames.constants import *
from onsets_and_frames.paths import *
from datetime import datetime

from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment('train_transcriber')


def _gpu_total_memory(use_gpu):
    gpu_total_memory = 0

    for i in use_gpu:
        gpu_total_memory += (torch.cuda
                             .get_device_properties(torch.device('cuda', i))
                             .total_memory)

    return gpu_total_memory


@ex.config
def config():
    logdir = LOG_DIR_PRE + datetime.now().strftime('%y%m%d-%H%M%S')
    # logdir = '/home/svcapp/userdata/amt_model/200804-172517'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 3000

    batch_size = 32
    sequence_length = 327680
    model_complexity_conv = 48
    model_complexity_lstm = 48
    use_gpu = [0]  # GPU Num in list

    dataset_list = ['MAESTRO']
    # dataset_list = ['MAPS']
    valid_dataset_list = ['MAESTRO']

    if torch.cuda.is_available():
        gpu_total_memory = _gpu_total_memory(use_gpu)
        if gpu_total_memory < 10e9:
            batch_size //= 2
            sequence_length //= 2
            print(('Reducing batch size to {} and sequence_length to '
                   '{} to save memory').format(batch_size, sequence_length))

    learning_rate = 0.0006
    learning_rate_decay_steps = 5000
    learning_rate_decay_rate = 0.95

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    # config for adversarial loss
    pix2pix_weight = 0  # if 0, don't use adversarial loss
    d_learning_rate = 0.0001

    # config for mixup
    mixup_strength = 0  # if 0, don't use mixup

    # config for meta-learner
    worker_id = None

    ex.observers.append(FileStorageObserver.create(logdir))
