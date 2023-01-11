from functools import wraps
import torch
from torch import nn

from models.force_estimator_transformers import Attention, AdaptiveTokenSampling
from typing import Tuple

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.attn_recordings = []
        self.samp_recordings = []
        self.attn_hooks = []
        self.samp_hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _attn_hook(self, _, input, output):
        self.attn_recordings.append(output.clone().detach())
    def _samp_hook(self, _, input, output):
        # Take the sample id which is the 3rd value on the sampling layer
        self.samp_recordings.append(output[2].clone().detach())

    def _register_hook(self):
        attn_modules = find_modules(self.vit.transformer, Attention)

        for attn_module in attn_modules:
            attn_handle = attn_module.attend.register_forward_hook(self._attn_hook)
            samp_handle = attn_module.ats.register_forward_hook(self._samp_hook)
            self.attn_hooks.append(attn_handle)
            self.samp_hooks.append(samp_handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.attn_hooks:
            hook.remove()
        for hook in self.samp_hooks:
            hook.remove()
        self.attn_hooks.clear()
        self.samp_hooks.clear()
        return self.vit

    def clear(self):
        self.attn_recordings.clear()
        self.samp_recordings.clear()

    def forward(self, img: torch.Tensor, return_sampled_token_ids: bool = False, robot_state: torch.Tensor = None):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        if return_sampled_token_ids:
            pred, token_ids = self.vit(img, return_sampled_token_ids, robot_state)
        else:
            pred = self.vit(img, return_sampled_token_ids, robot_state)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        attn_recordings = tuple(map(lambda t: t.to(target_device), self.attn_recordings))
        samp_recordings = tuple(map(lambda t: t.to(target_device), self.samp_recordings))

        #attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        attns = [rec for rec in attn_recordings]
        if return_sampled_token_ids:
            token_ids = [rec[: , 1:] - 1 for rec in samp_recordings]
            return pred, token_ids, attns

        return pred, attns