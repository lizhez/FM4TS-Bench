import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/CALF")

from ts_benchmark.baselines.LLM.submodules.CALF.models import CALF

class CALFModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        # config.pred_len = config.horizon
        self.model = CALF.Model(config, device)
        
        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc)
        return output['outputs_time']
