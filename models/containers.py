from contextlib import contextmanager
from torch import nn
from utils.typing import *


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()

    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand([batch_size, ] + list(self._buffers[name].shape[1:]))
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()


class ModuleList(nn.ModuleList, Module):
    pass


class ModuleDict(nn.ModuleDict, Module):
    pass
