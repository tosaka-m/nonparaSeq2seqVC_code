#coding:utf-8
import os
import os.path as osp
import re
import sys
import yaml
import shutil
from glob import glob
import numpy as np
import torch
import wandb
import click
from functools import reduce

from Utils.build_dataloader import build_dataloader
from Utils.build_optimizer import build_optimizer
from Utils.build_critic import build_critic
from Networks.build_model import build_model
from Utils.trainer import VCS2STrainer

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('-p', '--config_path', default='Configs/base_config.yml', type=str)
@click.option('-t', '--test', is_flag=True)
def main(config_path, test):
    config = yaml.safe_load(open('Configs/base_config.yml'))
    update_config = yaml.safe_load(open(config_path))
    config.update(update_config)
    train_config = config['train_config']
    log_dir = train_config['log_dir']

    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    if test:
        wandb.init(project="test", config=config)
    else:
        wandb.init(project="nonparaS2SVC", config=config)
        file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.addHandler(file_handler)

    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    device = train_config.get('device', 'cpu')
    save_freq = train_config.get('save_freq', 20)
    train_path = train_config.get('train_data', None)
    val_path = train_config.get('val_data', None)
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config=train_config.get('dataset_params', {}),
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config=train_config.get('dataset_params', {}))

    model = build_model(model_params=config['model_params'] or {})
    model.to(device)
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 5e-4)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    logger.info('scheduler_params :%s' % str(scheduler_params))
    optimizer = build_optimizer(
        {
            "main": reduce(lambda x, y: x+y,
                           map(list, [
                               model.text_encoder.parameters(),
                               model.audio_seq2seq.parameters(),
                               model.merge_net.parameters(),
                               model.speaker_encoder.parameters(),
                               model.decoder.parameters(),
                               model.postnet.parameters()])),
            "speaker_clf": model.speaker_classifier.parameters()
        },
        optimizer_params={}, scheduler_params=scheduler_params)


    critic = build_critic(critic_params={'vcs2s':{'n_speakers': config['model_params'].get('n_speakers', 200)}})
    trainer = VCS2STrainer(model=model,
                           critic=critic,
                           optimizer=optimizer,
                           device=device,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           logger=logger)

    if train_config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(
            train_config['pretrained_model'],
            train_config.get('load_only_params', False),
            train_config.get('load_scheduler', False))

    for epoch in range(1, epochs+1):
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.5f' % (key, value))
            else:
                results[key] = [wandb.Image(v) for v in value]
        wandb.log(results)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0



def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/nospace/train_list.txt"
    if val_path is None:
        val_path = "Data/nospace/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    # train_list = train_list[:100]
    # val_list = train_list[:100]
    return train_list, val_list

if __name__=="__main__":
    main()
