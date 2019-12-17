import torch
from torch import distributions
import utils
from models.containers import Module
from models.beam_search import *


class CaptioningModel(Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    def test(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, **kwargs) -> utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        device = utils.get_device(visual)
        outputs = []
        log_probs = []

        mask = torch.ones((b_s,), device=device)
        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                log_probs_t = self.step(t, out, visual, None, mode='feedback', **kwargs)
                out = torch.max(log_probs_t, -1)[1]
                mask = mask * (out.squeeze(-1) != eos_idx).float()
                log_probs.append(log_probs_t * mask.unsqueeze(-1).unsqueeze(-1))
                outputs.append(out)

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def sample_rl(self, visual: utils.TensorOrSequence, max_len: int, **kwargs) -> utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        outputs = []
        log_probs = []

        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                out = self.step(t, out, visual, None, mode='feedback', **kwargs)
                distr = distributions.Categorical(logits=out[:, 0])
                out = distr.sample().unsqueeze(1)
                outputs.append(out)
                log_probs.append(distr.log_prob(out).unsqueeze(1))

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def beam_search(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(visual, out_size, return_probs, **kwargs)
