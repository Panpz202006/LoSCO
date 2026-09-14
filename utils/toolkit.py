import os
import sys
import torch
import random
import logging
import numpy as np


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_logdir(args):
    logdir = 'logs/{}/{}/t{}'.format(args['method'], args['dataset'], args['sessions'])
    if args['debug']:
        logdir = os.path.join(logdir, 'debug')
    makedirs(logdir)
    return logdir


def setup_logging(logfilename, save_ckp=False):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if save_ckp:
        makedirs(logfilename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)])


def set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus


def set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))


def format_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return '{}h {}m {}s'.format(hours, minutes, seconds)


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def print_trainable_params(model, show_shapes=True):
    total_params = 0
    trainable_params = 0

    print("Parameters to be updated:")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
            if show_shapes:
                print(f"[Trainable] {name:60s} {tuple(param.shape)} | {num_params}")
            else:
                print(f"[Trainable] {name:60s} | {num_params}")

    logging.info(f"Total params:     {total_params:,}")
    logging.info(f"Trainable params: {trainable_params:,}")
    logging.info(f"Trainable ratio:  {100 * trainable_params / total_params:.4f}%")


def check_params_consistency(model, optimizer):
    model_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    model_param_set = set(model_params.values())

    optim_param_set = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optim_param_set.add(p)

    only_in_model = model_param_set - optim_param_set
    only_in_optim = optim_param_set - model_param_set
    ok = (len(only_in_model) == 0 and len(only_in_optim) == 0)

    if ok:
        print("✅ Requires_grad parameters and optimizer parameters are consistent.")
    else:
        print("❌ WARNING: Inconsistency detected!")

    return ok


def accuracy(y_pred, y_true, known_classes, increment=10):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    assert len(y_pred) == len(y_true), 'Data length error.'

    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy by task
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < known_classes)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= known_classes)[0] 
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def accuracy_all(y_pred, y_true):
    assert len(y_pred) == len(y_true), 'Data length error.'

    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()

    correct = (y_pred == y_true).sum()
    all_acc = np.around(correct * 100 / len(y_true), decimals=2)
    return all_acc
