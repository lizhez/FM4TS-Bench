import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules')


from torch import nn
from typing import Union
from einops import rearrange, repeat 
# from ts_benchmark.baselines.pre_train.model.Moirai_backbone import MoiraiBackbone
# from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from ts_benchmark.baselines.pre_train.submodules.uni2ts.model.moirai import MoiraiModule, MoiraiForecast
import torch

class Moirai(nn.Module):
    def __init__(
        self,
        config,
        variate_mode: str = 'M',
        patch_size: Union[str, int] = 64,
        model_size: str = 'base',
        scaling: bool = True,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.num_samples = config.num_samples
        self.patch_size = int(config.patch_size)
        
        # Load pretrained model
        self.moirai = MoiraiForecast(
            # module=MoiraiModule.from_pretrained(f"ts_benchmark/baselines/pre_train/checkpoints/moirai-lager"),
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=self.target_dim,
            num_samples=self.num_samples,
            past_feat_dynamic_real_dim=0,
            feat_dynamic_real_dim=0
            # scaling=scaling
        )
        if not config.use_p:
            for param in self.moirai.parameters():
                param.data.uniform_(-0.02, 0.02)

    def forward(self, inputs, dec_inp=None, x_mark_enc=None, x_mark_dec=None, device=None, num_samples=None):  
        # B, _, K = inputs.shape
        # inputs = rearrange(inputs, 'b l k -> (b k) l 1') #(210,96,1) need input (B, L, K)
        # past_observed_values = torch.tensor(torch.ones_like(inputs), dtype=torch.float64).to(device)
        past_is_pad = torch.tensor(torch.zeros(inputs.shape[:2]), dtype=torch.float64).to(device)     
        
        past_observed_target = torch.ones_like(inputs, dtype=torch.bool)
        # past_is_pad = torch.zeros_like(inputs, dtype=torch.bool).squeeze(-1)
        
        forecasts = self.moirai(
            past_target=inputs,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            num_samples=self.num_samples
        )
        # forecast = forecasts.mean(axis=1)
        forecast, _ = torch.median(forecasts, dim=1)

        # forecast = rearrange(forecast, '(b k) l -> b l k', b=B, k=K)
        return forecast
