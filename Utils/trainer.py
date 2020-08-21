import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import seaborn as sns
import numpy as np
import soundfile as sf
import torch
from torch import nn
from PIL import Image

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import matplotlib.pyplot as plt


matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 model,
                 critic=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.critic = critic
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger

    def _train_epoch(self):
        """Train model one epoch."""
        raise NotImplementedError

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    @torch.no_grad()
    def _get_images(self, **kwargs):
        return {}

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False, load_scheduler=True):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            if load_scheduler:
                state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

class VCS2STrainer(Trainer):
    def _train_epoch(self):
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            self.optimizer.zero_grad()
            batch = [b.to(self.device) for b in batch]
            text, text_lengths, mel_input, mel_target, mel_target_lengths, speaker_ids = batch
            output = self.model(text, text_lengths, mel_input, mel_target_lengths,
                                auto_encoding=(train_steps_per_epoch % 2 == 0))

            losses = self.critic['vcs2s'](output, text, text_lengths, mel_target, mel_target_lengths, speaker_ids)

            loss = 0
            for key, value in losses.items():
                loss += value
                train_losses['train/%s' % key].append(value.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        current_lr = self._get_lr()
        train_losses['train/learning_rate'] = current_lr
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device) for b in batch]
            text, text_lengths, _, mel_target, mel_target_lengths, speaker_ids = batch
            output = self.model(text, text_lengths, mel_target, mel_target_lengths,
                                auto_encoding=(eval_steps_per_epoch % 2 == 0))

            losses = self.critic['vcs2s'](output, text, text_lengths, mel_target, mel_target_lengths, speaker_ids)
            loss = 0
            for key, value in losses.items():
                loss += value
                eval_losses['eval/%s' % key].append(value.item())

            if eval_steps_per_epoch == 1:
                infered_data = self.model.inference(
                    mel_source=mel_target[:1],
                    mel_source_lengths=mel_target_lengths[:1],
                    mel_reference=mel_target[-1:])

                eval_images["eval/post_output"].append(
                    self.get_image([
                        output['post_output'][0].cpu().numpy(),
                        mel_target[0].cpu().numpy(),
                    ]))
                eval_images["eval/attns"].append(
                    self.get_image([
                        output['audio_seq2seq_alignments'][0].cpu().numpy(),
                        output['alignments'][0].cpu().numpy().T,
                    ]))
                eval_images["eval/inference"].append(
                    self.get_image([
                        infered_data['post_output'][0].cpu().numpy(),
                        mel_target[0].cpu().numpy(),
                        mel_target[-1].cpu().numpy(),
                        infered_data['alignment'][0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses

    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette
