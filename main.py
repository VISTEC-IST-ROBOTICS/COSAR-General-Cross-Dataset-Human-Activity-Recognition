# main.py

import os
import argparse
import numpy as np
import torch
from copy import deepcopy
from typing import Literal
from itertools import product

import configure
from data.loader import DataLoader
from data.preprocessing import Preprocessing, SlidingWindow, PGBA
from data.management import build_dataset, CustomDataset
from utils import set_seed, calculate_sd_pooled, print_head_border
from learner import create_leanner
from utils import get_confusion_matrix


def get_arguments():
    parser = argparse.ArgumentParser(description='Attention Based Sensor Fusion')

    parser.add_argument('-n', '--device',     type=str, default='cuda:0',      help='device to use (default: cuda:0)')
    parser.add_argument('-d', '--dataset',    type=str, default='pamap2',      help='dataset name (default: pamap2)')
    parser.add_argument('-f', '--model-file', type=str, default='',            help='model file path to load')
    parser.add_argument('--model',            type=str, default='cosar',       help='model to train (default: cosar)')
    parser.add_argument('--fe',               type=str, default='aug-cwt',     help='feature extraction method (default: aug-cwt)')
    parser.add_argument('--method',           type=str, default='supervised',  help='learning method (default: supervised)')

    parser.add_argument('--all',        action='store_true', help='use all activities')
    parser.add_argument('--training',   action='store_true', help='training mode')
    parser.add_argument('--testing',    action='store_true', help='testing mode')
    parser.add_argument('--seed_eval',  action='store_true', help='evaluate across multiple seeds')
    parser.add_argument('--kde_eval',   action='store_true', help='evaluate KDE parameter sensitivity')
    parser.add_argument('--ft-config',  action='store_true', help='use fine-tuning configuration')
    parser.add_argument('--log-wandb',  action='store_true', help='log to Weights & Biases')

    return parser.parse_args()


def run_transforms(raw_dataset, sampling, data_config: dict,
                   feature_extraction: Literal['AugmentedCWT', 'AugmentedSTFT'] = None, verbose=False):
    """
    Return
        data shape will be [n_samples, n_sensors, n_channels, ...] after all transforms.
        If the feature extraction is applied, the last two dims will be [n_freqs, n_frames] for AugmentedSTFT or [n_scales, time_resize] for AugmentedCWT.
        In this case, the output of
        Augmented STFT:    [20, 11]
        Augmented CWT:     [24, 75]
    """
    dataset = deepcopy(raw_dataset)

    from data import preprocessing as preprocessing_module
    feature_cls = getattr(preprocessing_module, feature_extraction) if feature_extraction else None

    transforms = [
        Preprocessing(dataset_sampling=dataset['fs'], desired_sampling=sampling),
        PGBA(**data_config.get('kde_params', {})) if data_config.get('is_rotate') else None,
        SlidingWindow(sampling=sampling, **data_config.get('segment', {})),
        feature_cls(sampling=sampling)            if feature_cls                  else None,
    ]

    for transform in filter(None, transforms):
        dataset = transform(dataset)

    return dataset


def evaluate_model(learner, test_data, subject_sample_count, dataset, pretrained_params, 
                   verbose=False, get_confusion=False):
    """
    Evaluate a trained model on per-subject test subsets.

    Args:
        learner:                A Learning instance (e.g. SupervisedLearning).
        test_data:              (X_test, y_test) tensors for the full test split.
        subject_sample_count:   {subject_id: n_samples} ordered dict.
        dataset:                Must contain key 'classes' (list of class names).
        pretrained_params:      State dict to load into the model before evaluation.
        verbose:                Print per-subject acc/F1 if True. Default: False.
        get_confusion:          If True, also return (y_preds, y_trues) tensors
               of shape [total_samples] for multi-seed CM stacking

    Returns:
        acc_stats:  (mean_acc, std_acc) across subjects in %.
        f1_stats:   (mean_f1,  std_f1)  across subjects in %.
        labels:     (y_trues, y_preds) tuple of tensors for confusion matrix calculation if seed_evals=True, else None.
    """
    current_idx = 0
    acc_list, f1_list = [], []
    y_preds_all, y_trues_all = [], []

    for s in subject_sample_count.keys():
        n_sample    = subject_sample_count[s]
        x_test      = test_data[0][current_idx:current_idx + n_sample]
        y_test      = test_data[1][current_idx:current_idx + n_sample]
        test_subset = CustomDataset(x_test, y_test, use_transform=False)
        current_idx += n_sample

        acc, f1, _ = learner.test(test_subset,
            name_classes=dataset['classes'],
            params=pretrained_params,
            confusion_matrix=False
        )
        acc_list.append(acc)
        f1_list.append(f1.to('cpu'))

        if get_confusion:
            y_trues = learner.y_trues_finish.to('cpu')
            y_preds = learner.y_preds_finish.to('cpu')
            
            y_trues_all.append(y_trues)
            y_preds_all.append(y_preds)

        if verbose:
            print(f'    Subject {s}: acc={acc:.2f}%, f1={f1:.2f}%')
            if acc < 70:
                conf = get_confusion_matrix(y_preds, y_trues, tasks=dataset['classes'], 
                                normalize='true', verbose=verbose
                                )

    acc_stats = (np.mean(acc_list), np.std(acc_list))
    f1_stats  = (np.mean(f1_list),  np.std(f1_list))

    if get_confusion:
        y_preds = torch.cat(y_preds_all, dim=0)
        y_trues = torch.cat(y_trues_all, dim=0)
        outputs = [y_preds, y_trues]
        
        return acc_stats, f1_stats, outputs

    return acc_stats, f1_stats, None


if __name__ == "__main__":

    path      = os.getcwd()
    arguments = get_arguments()
    
    print(arguments)

    # Configuration
    default_seed    = configure.seed
    sensor_types    = configure.sensor_types
    positions       = configure.positions
    sampling        = configure.sampling
    is_rotate       = configure.is_rotate
    # If none of the specific conditions are align_tag, it will default to an empty string (no subfolder).
    align_tag       = 're' if is_rotate else 'naive'
    b, h = configure.b, configure.h
    
    # Arguments from command line will override the default configuration
    model_file      = arguments.model_file
    dataset_name    = arguments.dataset

    # Resolve model and feature extraction
    model_priority = [
        ('bams',     'bams'     in arguments.model or 'bams'     in model_file),
        ('baseline', 'baseline' in arguments.model or 'baseline' in model_file),
        ('llm4har',  'llm4har'  in arguments.model or 'llm4har'  in model_file),
        ('cosar',    True),
    ]
    feature_priority = [
        (None,           'baseline' in arguments.model or 'baseline' in model_file),
        (None,           'llm4har'  in arguments.model or 'llm4har' in model_file),
        ('AugmentedSTFT','aug-stft' in arguments.fe or 'stft' in model_file),
        ('AugmentedCWT', True),
    ]

    model   = next(name for name, condition in model_priority   if condition)
    feature = next(name for name, condition in feature_priority if condition)

    desc = f'Feature extraction: {feature} | Model: {model}'
    print(); print_head_border(desc, length=len(desc) + 10); print()

    # Resolve seeds
    if '.pt' in model_file:
        # Get the number of seed if model file contains the number
        seeds = [int(model_file.split("-")[-1].split(".")[0])]
        if not seeds:
            seeds = [0]
    else:
        if 'seed' in model_file or arguments.seed_eval:
            seeds = range(5)
        else:
            seeds = [0]

    # Loading is deterministic (interpolate only) — no seed needed here
    set_seed(default_seed)
    downloader  = DataLoader()
    raw_dataset = downloader.load(dataset_name, 'custom', positions, sensor_types=sensor_types)
    
    data_config = {
        'segment':    {'window': configure.window_size, 'step': configure.step_size[dataset_name]},
        'is_rotate':   is_rotate,
        'kde_params': {'b': b, 'h': h}
    }


    # =====================================================================
    # KDE Sensitivity Evaluation
    # =====================================================================
    if arguments.kde_eval or 'kde' in model_file:
        b_values = range(10, 110, 10)
        h_values = [round(h * 0.05, 2) for h in range(1, 11)]
        acc_results = []
        combinations = []

        for b, h in product(b_values, h_values):
            # Override the data_config for each combination of b and h
            data_config = {
                'segment':   {'window': configure.window_size, 'step': configure.step_size[dataset_name]},
                'is_rotate':  True,
                'kde_params': {'b': b, 'h': h}
            }
            set_seed(default_seed)
            dataset = run_transforms(raw_dataset, sampling, data_config, feature_extraction=feature)

            if arguments.training:
                model_file_name = f'{model}/kde/{dataset_name}/b={b},h={h}.pt'
                
                train_data, test_data, _ = build_dataset(dataset, val_split=0.2, seed=default_seed)

                feature_size     = train_data[0].shape[-2:]
                train_dataset    = CustomDataset(*train_data, use_transform=True, num_masks=2)
                val_dataset      = CustomDataset(*test_data,  use_transform=False)
                hyperparams      = configure.hyperparams_config[model]

                learner = create_leanner(len(dataset['classes']), feature_size, model=model, device=arguments.device, seed=0, **hyperparams)
                learner.train(train_dataset, val_dataset, **hyperparams)
                learner.save_model(path=path, filename=model_file_name)
            else:
                model_file_name = f'{model_file}/b={b},h={h}.pt'
                print(model_file_name)
                _, test_data, subject_sample_count = build_dataset(dataset, stage='test', seed=default_seed)
                feature_size      = test_data[0].shape[-2:]
                pretrained_params = torch.load(model_file_name, weights_only=True)
                learner           = create_leanner(len(dataset['classes']), feature_size, model=model, device=arguments.device)

                (acc, acc_sd), (f1, f1_sd), _ = evaluate_model(learner, test_data, subject_sample_count, dataset, pretrained_params)
                acc_results.append(round(float(acc), 2))
                combinations.append((h, b))
        
        if not arguments.training:
            idx = 0
            for h in h_values:
                lst = []
                for b in b_values:
                    lst.append(acc_results[idx])
                    idx += 1
                print(lst)
            
    # =====================================================================
    # Training
    # =====================================================================
    elif arguments.training:
        
        subfolder_name = [
            ('stft', arguments.fe == 'aug-stft'),
            ('seed', arguments.seed_eval),
            ('', True)
        ]
        # We create the subfolder name based on the conditions, 
        # but if none of the specific conditions are align_tag, it will default to an empty string (no subfolder).
        sf_name = next(name for name, condition in subfolder_name if condition)
        
        set_seed(default_seed)
        dataset = run_transforms(raw_dataset, sampling, data_config, feature_extraction=feature)
        
        for idx, exp_seed in enumerate(seeds):
            set_seed(exp_seed)
            print(f'Random seed set to: {exp_seed}')
            
            train_data, test_data, _ = build_dataset(dataset, val_split=0.2, seed=exp_seed)
            feature_size = train_data[0].shape[-2:]
            hyperparams  = configure.hyperparams_config[model]

            train_dataset = CustomDataset(*train_data, use_transform=True, num_masks=2)
            val_dataset   = CustomDataset(*test_data,  use_transform=False)

            if idx == 0:
                print(f'Training the model on shape {train_dataset.X_train.shape}')
            
            if sf_name == '':
                model_file_name = f'{model}/{dataset_name}-{align_tag}.pt'
            else:
                model_file_name = f'{model}/{sf_name}/{dataset_name}/{align_tag}-{exp_seed}.pt'
                
            # Create the learner (model) and train the model
            learner = create_leanner(len(dataset['classes']), feature_size, model=model, device=arguments.device, seed=exp_seed, **hyperparams)
            learner.train(train_dataset, val_dataset, **hyperparams)
            learner.save_model(path=path, filename=model_file_name)

    # =====================================================================
    # Testing
    # =====================================================================
    else:

        set_seed(default_seed)
        dataset = run_transforms(raw_dataset, sampling, data_config, feature_extraction=feature)

        print(); print_head_border('Evaluating the model')
        np.set_printoptions(precision=4, suppress=True)

        acc_list, acc_sd_list = [], []
        f1_list,  f1_sd_list  = [], []
        all_conf = []
        
        get_confusion = True

        for exp_seed in seeds:
            set_seed(exp_seed)
            _, test_data, subject_sample_count = build_dataset(dataset, stage='test', seed=exp_seed)
            feature_size = test_data[0].shape[-2:]
            n_subjects   = len(subject_sample_count)
            total_sample = sum(subject_sample_count.values())
            
            if '.pt' not in model_file:
                file = f'{model_file}/{align_tag}-{exp_seed}.pt'
            else:
                file = model_file
                
            pretrained_params = torch.load(file, weights_only=True)
            learner           = create_leanner(len(dataset['classes']), feature_size, model=model, device=arguments.device)

            # Verbose for single seed evaluation to see per-subject performance, 
            # otherwise just print the average across subjects for multiple seeds
            verbose = len(seeds) == 1
            acc_stats, f1_stats, labels = evaluate_model(learner, test_data, subject_sample_count, dataset, 
                                                        pretrained_params, verbose=verbose, get_confusion=get_confusion
                                                        )
            desc = f'acc={acc_stats[0]:.2f}±{acc_stats[1]:.2f}%, f1={f1_stats[0]:.2f}±{f1_stats[1]:.2f}%'
            print(f'[seed={exp_seed}] {n_subjects} subjects (n={total_sample}): {desc}')

            acc_list.append(acc_stats[0]);  acc_sd_list.append(acc_stats[1])
            f1_list.append(f1_stats[0]); f1_sd_list.append(f1_stats[1])
            
            if get_confusion:
                y_preds, y_trues = labels[0], labels[1]
                conf = get_confusion_matrix(y_preds, y_trues, tasks=dataset['classes'], 
                                normalize='true', verbose=False
                                )
                all_conf.append(conf)

        Ns            = [n_subjects] * len(seeds)
        acc_sd_pooled = calculate_sd_pooled(acc_sd_list, Ns)
        f1_sd_pooled  = calculate_sd_pooled(f1_sd_list,  Ns)

        summary = f'Accuracy={np.mean(acc_list):.2f}±{acc_sd_pooled:.2f}%, F1={np.mean(f1_list):.2f}±{f1_sd_pooled:.2f}%'
        print_head_border(f'Average across {len(seeds)} seeds: {summary}', color='green')
        
        if get_confusion:
            avg_conf = np.mean(all_conf, axis=0)
            print('Average Confusion Matrix across seeds:')
            print(avg_conf)
