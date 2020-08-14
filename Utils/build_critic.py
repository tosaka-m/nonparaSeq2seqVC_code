#coding:utf-8
import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import yaml
import torch.nn.functional as F

def build_critic(critic_params={}):
    critic = {
        "mse": nn.MSELoss(),
        "masked_mse": MaskedMSELoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "masked_bce": MaskedBCEWithLogitsLoss(),
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "masked_ce": MaskedCrossEntropyLoss(ignore_index=-1),
        "ga": GuidedAttentionLoss(alpha=1000),
        "ctc": torch.nn.CTCLoss(),
        "mle": MLE(**critic_params.get('mle', {})),
        "vcs2s": VCS2SLoss()
    }
    return critic

class VCS2SLoss(nn.Module):
    def __init__(self):
        super(VCS2SLoss, self).__init__()
        self.masked_mse = MaskedMSELoss()
        self.masked_bce = MaskedBCEWithLogitsLoss()
        self.masked_ce = MaskedCrossEntropyLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.constractive = ConstrastiveLoss()

    def forward(self, output, text_input, text_input_lengths, mel_target, mel_target_lengths, speaker_ids):
        text_mask = self.get_mask_from_lengths(text_input_lengths)
        mel_mask = self.get_mask_from_lengths(mel_target_lengths)

        # taco losses
        B = text_input.size(0)
        recon_loss = self.masked_mse(output['predict_mel'], mel_target, ~mel_mask.unsqueeze(1))
        recon_post_loss = self.masked_mse(output['post_output'], mel_target, ~mel_mask.unsqueeze(1))
        stop_loss = self.masked_bce(output['predicted_stop'],
                                    torch.zeros(B, mel_mask.size(1)//2), ~mel_mask.reshape(B, -1, 2)[:,:,0])

        # spk losses
        spk_logit = output['speaker_logit_from_mel']
        spk_enc_loss = self.ce(spk_logit, speaker_ids)

        spk_logit_hidden = output['speaker_logit_from_mel_hidden']
        spk_target_flatten = speaker_ids.unsqueeze(1).expand(-1, spk_logit_hidden.size(1)).reshape(-1)
        spk_logit_flatten = spk_logit_hidden.reshape(-1, spk_logit_hidden.size(2))
        spk_clf_loss = self.ce(spk_logit_flatten, spk_target_flatten)

        n_spk = spk_logit.size(1)
        flatten_mask =  text_mask.unsqueeze(2).expand(-1,-1, n_spk).reshape(-1, n_spk)
        flatten_target = (1/n_spk) * torch.ones_like(spk_logit_flatten)
        spk_adv_loss = self.masked_mse(
            spk_logit_flatten.softmax(dim=1).unsqueeze(1),
            flatten_target.unsqueeze(1),
            flatten_mask.unsqueeze(1))

        # text losses
        text_logit = output['audio_seq2seq_logit'][:, :-1]
        text_logit_flatten = text_logit.reshape(-1, text_logit.size(2))
        text_target_flatten = text_input.reshape(-1)
        text_clf_loss = self.masked_ce(text_logit_flatten, text_target_flatten, ~text_mask.reshape(-1))
        return {
            "recon_loss": recon_loss,
            "recon_post_loss": recon_post_loss,
            "stop_loss": stop_loss,
            "spk_enc_loss": spk_enc_loss,
            "spk_clf_loss": spk_clf_loss,
            "spk_adv_loss": spk_adv_loss,
            "text_clf_loss": text_clf_loss,
        }


    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths.device)
        #print ids
        mask = (ids < lengths.unsqueeze(1))
        return mask


class MLE(nn.Module):
    def __init__(self, n_mels=80, n_squeeze=1):
        super(MLE, self).__init__()
        self.const = 0.5 * np.log(2 * np.pi)
        self.n_squeeze = n_squeeze
        self.n_mels = n_mels
        # self.weight = torch.load(osp.join(osp.diname(__file__), 'inv_freq_weight.pth'))
        # self.use_weight = False

    def forward(self, z, mean, logstd, logdet, lengths):
        mle = self.const + (torch.sum(logstd) + 0.5*torch.sum(torch.exp(-2*logstd)*(z-mean)**2) \
                            - torch.sum(logdet)) / (torch.sum(lengths // self.n_squeeze)*self.n_squeeze*self.n_mels)
        return mle


class MaskedMSELoss(nn.Module):
    def __init__(self, reduce='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduce = reduce

    def forward(self, x, y, mask):
        """
        x.shape = (B, H, L)
        y.shape = (B, H, L)
        mask.shape = (B, 1, L), True index will be ignored
        """
        assert(x.shape == y.shape)
        mask = mask.type_as(x)
        shape = x.shape
        lengths = (1 - mask).sum(dim=tuple(range(1, len(shape)))).detach()
        sq_error = ((x * (1 - mask) - y * (1 - mask))**2).mean(dim=-1).sum(dim=1)
        mean_sq_error = sq_error / lengths
        if self.reduce is not None:
            mean_sq_error = getattr(mean_sq_error, self.reduce)()
        return mean_sq_error

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduce='mean'):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduce = reduce

    def forward(self, x, y, mask):
        """
        x = pred
        y = target
        """
        assert(x.shape == y.shape)
        mask = mask.type_as(x)
        shape = x.shape
        lengths = (1 - mask).sum(dim=tuple(range(1, len(shape)))).detach()
        prob = torch.sigmoid(x)
        bce_loss = (-1)*((y*torch.log(prob) + (1-y)*torch.log(1-prob))*(1-mask)).sum(dim=tuple(range(1, len(shape))))
        bce_loss /= lengths
        if self.reduce is not None:
            bce_loss =getattr(bce_loss, self.reduce)()
        return bce_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, reduce='mean', **kwargs):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.reduce = reduce
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', **kwargs)

    def forward(self, x, y, mask):
        """
        x = pred
        y = target
        """
        mask = mask.type_as(x)
        ce_loss = self.ce(x, y)

        ce_loss *= (1-mask)
        ce_loss = ce_loss.sum() / (mask.sum())
        if self.reduce is not None:
            ce_loss =getattr(ce_loss, self.reduce)()
        return ce_loss


class ConstrastiveLoss(nn.Module):
    def __init__(self):
        super(ConstrastiveLoss, self).__init__()

    def forward(self, x, y, x_mask, y_mask):
        """
        x.shape = (B, Tx, H)
        y.shape = (B, Ty, H)
        x_mask = (B, Tx)
        y_mask = (B, Ty)
        """
        norm_x = x / x.norm(dim=2, keepdim=True)
        norm_y = y / y.norm(dim=2, keepdim=True)
        distance_matrix = 2 - 2 * torch.bmm(norm_x, norm_y.transpose(1, 2))
        constrast_mask = x_mask.unsqueeze(1) & y_mask.unsqueeze(2)
        hard_alignments = torch.eye(distance_matrix.size(1))
        constrast_loss = hard_alignments * distance_matrix + (1 - hard_alignments) * torch.max(1 - distance_matrix, torch.zeros_like(distance_matrix))
        constrast_loss = (constrast_loss * constrast_mask).sum() / constrast_mask.sum()
        return constrast_loss

class GuidedAttentionLoss(torch.nn.Module):

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):

        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
            not_through_mask_start = torch.arange(att_ws.shape[2]).reshape(1, 1, -1).expand_as(att_ws)\
                               <  torch.arange(att_ws.shape[1]).reshape(1, -1, 1).expand_as(att_ws)
            not_through_mask_start = not_through_mask_start.to(att_ws.device)
            not_through_mask_end = torch.arange(att_ws.shape[2]).reshape(1, 1, -1).expand_as(att_ws)\
                                   - torch.arange(att_ws.shape[1]).reshape(1, -1, 1).expand_as(att_ws)
            not_through_mask_end = not_through_mask_end.to(att_ws.device)
            not_through_mask_end = not_through_mask_end > (ilens.reshape(-1, 1, 1) -olens.reshape(-1, 1, 1))
            not_through_mask = not_through_mask_start | not_through_mask_end

            self.masks = self.masks | not_through_mask


        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)

def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
