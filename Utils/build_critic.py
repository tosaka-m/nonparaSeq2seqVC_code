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
        "vcs2s": VCS2SLoss(**critic_params.get('vcs2s', {}))
    }
    return critic

class VCS2SLoss(nn.Module):
    def __init__(self,
                 contra_w=30.,
                 text_clf_w=2.,
                 spk_enc_w=1.,
                 spk_clf_w=0.1,
                 spk_adv_w=100.,
                 smoothing=0.1,
                 n_tokens=41,
                 n_speakers=200,
    ):
        super(VCS2SLoss, self).__init__()
        self.masked_mse = MaskedMSELoss()
        self.masked_l1 = MaskedL1Loss()
        self.masked_bce = MaskedBCEWithLogitsLoss()
        self.masked_ce = MaskedCrossEntropyLoss(classes=n_tokens)
        self.ce = LabelSmoothingLoss(classes=n_speakers, smoothing=0.1) #nn.CrossEntropyLoss(ignore_index=-1)
        self.contrastive = ContrastiveLoss()

        self.contra_w = contra_w
        self.text_clf_w = text_clf_w
        self.spk_enc_w = spk_enc_w
        self.spk_clf_w = spk_clf_w
        self.spk_adv_w = spk_adv_w
        self.smoothing = 0.1
        self.eos = 2

    def forward(self, output, text_input, text_input_lengths, mel_target, mel_target_lengths, speaker_ids):
        """
        output:
        text_input:
        """
        device = text_input.device
        text_mask = self.get_mask_from_lengths(text_input_lengths)
        mel_mask = self.get_mask_from_lengths(mel_target_lengths)

        # taco losses
        B = text_input.size(0)
        recon_loss = self.masked_l1(output['predict_mel'], mel_target, ~mel_mask.unsqueeze(1))
        recon_post_loss = self.masked_l1(output['post_output'], mel_target, ~mel_mask.unsqueeze(1))
        stop_label = torch.zeros(B, mel_mask.size(1)//2).to(device)
        stop_label.scatter_(1, (mel_target_lengths // 2 -1).unsqueeze(1), 1)
        stop_label.scatter_(1, ((mel_target_lengths-1) // 2).unsqueeze(1), 1)
        stop_label = stop_label * (1 - 2 * self.smoothing) + self.smoothing
        stop_loss = self.masked_bce(output['predicted_stop'],
                                    stop_label, ~mel_mask.reshape(B, -1, 2)[:,:,0])

        # spk losses
        spk_logit = output['speaker_logit_from_mel']
        spk_enc_loss = self.ce(spk_logit, speaker_ids)
        spk_logit_hidden = output['speaker_logit_from_mel_hidden']
        spk_target_flatten = speaker_ids.unsqueeze(1).expand(-1, spk_logit_hidden.size(1)).reshape(-1)
        spk_logit_flatten = spk_logit_hidden.reshape(-1, spk_logit_hidden.size(2))
        spk_clf_loss = self.ce(spk_logit_flatten, spk_target_flatten)

        n_spk = spk_logit.size(1)
        expand_mask =  text_mask.unsqueeze(1).expand(-1, n_spk, -1)
        uniform_target = (1/n_spk) * torch.ones_like(spk_logit_hidden)
        spk_adv_loss = self.masked_mse(
            spk_logit_hidden.transpose(1, 2).softmax(dim=1),
            uniform_target.transpose(1, 2),
            ~expand_mask)

        # text losses
        text_logit = output['audio_seq2seq_logit']
        text_logit_flatten = text_logit.reshape(-1, text_logit.size(2))
        text_target = torch.cat([text_input, torch.zeros((text_input.size(0), 1)).type_as(text_input)], dim=1)
        text_target.scatter_(1, text_input_lengths.unsqueeze(1), self.eos)
        text_target_flatten = text_target.reshape(-1)
        text_target_mask = self.get_mask_from_lengths(text_input_lengths+1)
        text_clf_loss = self.masked_ce(text_logit_flatten, text_target_flatten, ~text_target_mask.reshape(-1))

        # contrastive loss
        text_hidden = output['text_hidden']
        mel_hidden = output['audio_seq2seq_hidden']
        contra_loss = self.contrastive(text_hidden, text_mask, mel_hidden, text_mask)

        return {
            "recon_loss": recon_loss,
            "recon_post_loss": recon_post_loss,
            "stop_loss": stop_loss,
            "spk_enc_loss": spk_enc_loss * self.spk_enc_w,
            "spk_clf_loss": spk_clf_loss * self.spk_clf_w,
            "spk_adv_loss": spk_adv_loss * self.spk_adv_w,
            "text_clf_loss": text_clf_loss * self.text_clf_w,
            "contra_loss": contra_loss * self.contra_w
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

class MaskedL1Loss(nn.Module):
    def __init__(self, reduce='mean'):
        super(MaskedL1Loss, self).__init__()
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
        sq_error = (x * (1 - mask) - y * (1 - mask)).abs().mean(dim=-1).sum(dim=1)
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
    def __init__(self, reduce='mean', classes=200, smoothing=0.1, **kwargs):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.reduce = reduce
        #self.ce = torch.nn.CrossEntropyLoss(reduction='none', **kwargs)
        self.ce = LabelSmoothingLoss(classes, smoothing, reduce=None)

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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, reduce='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduce = reduce

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * pred, dim=self.dim)
        if self.reduce is not None:
            loss = getattr(loss, self.reduce)()

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x, x_mask, y, y_mask):
        """
        x.shape = (B, Tx, H)
        y.shape = (B, Ty, H)
        x_mask = (B, Tx)
        y_mask = (B, Ty)
        """
        norm_x = x / (1e-8 + x.norm(dim=2, keepdim=True))
        norm_y = y / (1e-8 + y.norm(dim=2, keepdim=True))
        distance_matrix = 2 - 2 * torch.bmm(norm_x, norm_y.transpose(1, 2))
        contrast_mask = x_mask.unsqueeze(1) & y_mask.unsqueeze(2)
        hard_alignments = torch.eye(distance_matrix.size(1)).to(x.device)
        contrast_loss = hard_alignments * distance_matrix + (1 - hard_alignments) * torch.max(1 - distance_matrix, torch.zeros_like(distance_matrix))
        contrast_loss = (contrast_loss * contrast_mask).sum() / contrast_mask.sum()

        return contrast_loss

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
