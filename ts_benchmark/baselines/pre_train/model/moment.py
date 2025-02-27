import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/moment')

from einops import rearrange
from torch import nn

from ts_benchmark.baselines.pre_train.submodules.moment import MOMENTPipeline

class Moment(nn.Module):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset

        # self.no_training = True

        self.model = MOMENTPipeline.from_pretrained(
            "ts_benchmark/baselines/pre_train/checkpoints/MOMENT", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': config.pred_len,
                'seq_len': config.seq_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True, # Freeze the patch embedding layer
                'freeze_embedder': True, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
            },
        )
        self.model.init()
        if not config.use_p:
            for param in self.model.parameters():
                param.data.uniform_(-0.02, 0.02)
        

    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> b k l')
        point_forecast = self.model(x_enc=inputs)
        forecast = point_forecast.forecast
        output = rearrange(forecast, 'b k l -> b l k', b=B, k=K)
        return output
