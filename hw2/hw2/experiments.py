import argparse
import itertools
import os
import random
import sys
import json

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs236781.train_results import FitResult
from . import cnn
from . import training

DATA_DIR = os.path.expanduser('~/.pytorch-datasets')

MODEL_TYPES = dict(cnn=cnn.ConvClassifier,
                   resnet=cnn.ResNetClassifier,
                   ycn=cnn.YourCodeNet)


def run_experiment(run_name, out_dir='./results', seed=None, device=None,
                   # Training params
                   bs_train=128, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, lr=1e-3, reg=1e-3,
                   # Model params
                   filters_per_layer=[64], layers_per_block=2, pool_every=2,
                   hidden_dims=[1024], model_type='cnn',
                   **kw):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    x0, _ = ds_train[0]
    in_size = x0.shape
    num_classes = 10
    filters = [layer for layer in filters_per_layer for _ in range(layers_per_block)]
    model = model_cls(in_size=in_size, out_classes=num_classes, channels=filters,
                      pool_every=pool_every, hidden_dims=hidden_dims).to(device)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg,)
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    trainer = training.TorchTrainer(model, loss_fn, optimizer, device)

    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False)
    fit_tmp = trainer.fit(dl_train, dl_test, num_epochs=epochs, checkpoints='./checkpoint',
                          early_stopping=early_stopping, max_batches=batches)
    train_loss_list = [tr_loss.item() for tr_loss in fit_tmp.train_loss]
    train_acc_list = [tr_acc.item() for tr_acc in fit_tmp.train_acc]
    test_loss_list = [tst_loss.item() for tst_loss in fit_tmp.test_loss]
    test_acc_list = [tst_acc.item() for tst_acc in fit_tmp.test_acc]
    fit_res = FitResult(num_epochs=fit_tmp.num_epochs, train_loss=train_loss_list,
             train_acc=train_acc_list, test_loss=test_loss_list,
             test_acc=test_acc_list)
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)

def experiment1_1():
    K = [32, 64]
    L = [2, 4, 8, 16]
    for k in K:
        for l in L:
            torch.cuda.empty_cache()
            exp_name = 'exp1_1'
            run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=[k], layers_per_block=l, pool_every=4, hidden_dims=[100],
                          model_type='cnn',)


def experiment1_2():
    L = [2, 4, 8]
    K = [32, 64, 128, 256]
    for l in L:
        for k in K:
            torch.cuda.empty_cache()
            exp_name = 'exp1_2'
            run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=[k], layers_per_block=l, pool_every=4, hidden_dims=[100],
                          model_type='cnn',)


def experiment1_3():
    L = [1, 2, 3, 4]
    K = [64, 128, 256]
    for l in L:
        torch.cuda.empty_cache()
        exp_name = 'exp1_3'
        run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=K, layers_per_block=l, pool_every=4, hidden_dims=[100],
                          model_type='cnn',)

def experiment1_4():
    K1 = [32]
    L1 = [8, 16, 32]
    K2 = [64, 128, 256]
    L2 = [2, 4, 8]
    for k in K1:
        for l in L1:
            torch.cuda.empty_cache()
            exp_name = 'exp1_4'
            run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=[k], layers_per_block=l, pool_every=6, hidden_dims=[100],
                          model_type='resnet',)
    for l in L2:
        torch.cuda.empty_cache()
        exp_name = 'exp1_4'
        run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=K2, layers_per_block=l, pool_every=6, hidden_dims=[100],
                          model_type='resnet',)
        
def experiment2():
    L = [3, 6, 9, 12]
    K = [32, 64, 128]
    for l in L:
        torch.cuda.empty_cache()
        exp_name = 'exp2'
        run_experiment(exp_name, seed=42, bs_train=100, batches=500, epochs=20, early_stopping=5,
                           filters_per_layer=K, layers_per_block=l, pool_every=8, hidden_dims=[100],
                          model_type='ycn',)
        
def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(
        config=cfg,
        results=fit_res._asdict()
    )

    cfg_LK = f'L{cfg["layers_per_block"]}_K' \
             f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    output_filename = f'{os.path.join(out_dir, run_name)}_{cfg_LK}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = FitResult(**output['results'])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description='CS236781 HW2 Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-exp', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument('--run-name', '-n', type=str,
                        help='Name of run and output file', required=True)
    sp_exp.add_argument('--out-dir', '-o', type=str, help='Output folder',
                        default='./results', required=False)
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)
    sp_exp.add_argument('--device', '-d', type=str,
                        help='Device (default is autodetect)',
                        default=None, required=False)

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=100)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=100)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=int,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', type=float,
                        help='Learning rate', default=1e-3)
    sp_exp.add_argument('--reg', type=float,
                        help='L2 regularization', default=1e-3)

    # # Model
    sp_exp.add_argument('--filters-per-layer', '-K', type=int, nargs='+',
                        help='Number of filters per conv layer in a block',
                        metavar='K', required=True)
    sp_exp.add_argument('--layers-per-block', '-L', type=int, metavar='L',
                        help='Number of layers in each block', required=True)
    sp_exp.add_argument('--pool-every', '-P', type=int, metavar='P',
                        help='Pool after this number of conv layers',
                        required=True)
    sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
                        help='Output size of hidden linear layers',
                        metavar='H', required=True)
    sp_exp.add_argument('--model-type', '-M',
                        choices=MODEL_TYPES.keys(),
                        default='cnn', help='Which model instance to create')

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == '__main__':
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
