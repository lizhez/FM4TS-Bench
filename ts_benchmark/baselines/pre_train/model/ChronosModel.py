# ---------------------------------------------------------------------------------
# Portions of this file are derived from Chronos
# - Source: https://github.com/amazon-science/chronos-forecasting
# - Paper: Chronos: Learning the Language of Time Series
# - License: Apache License 2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------
import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/chronos')
import torch
from ts_benchmark.baselines.pre_train.submodules.chronos import ChronosBoltPipeline
from einops import rearrange
from torch import nn

class Chronos(nn.Module):
    def __init__(
        self,
        config,
        model_size: str = 'base',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # self.pred_len = kwargs.get('prediction_length')
        
        # if type(self.prediction_length) == list:
        #     self.prediction_length = max(self.prediction_length)
            

        # if type(self.context_length) == list:
        #     self.context_length = max(self.context_length)
        
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.pred_len = self.prediction_length

        # Load pretrained model
        # self.no_training = True
        # Load Chronos
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            # "amazon/chronos-t5-{}".format(model_size),
            "ts_benchmark/baselines/pre_train/checkpoints/chronos",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        if not config.use_p:
            for param in self.pipeline.model.parameters():
                param.data.uniform_(-0.02, 0.02)

    # def forward(self, inputs, dec_input, input_mark, target_mark, device, input_stamp_np, num_samples=100):
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec, device=None, num_samples=None):  
        # inputs = self.get_inputs(batch_data, 'encode')
        # inputs = inputs[:, -self.context_length:]
        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l').cpu()
        context = [inputs[i] for i in range(B*K)]
        inner_batch_size = 12 # for 80G gpu
        forecast_samples = []

        # Process in batches of size `inner_batch_size`
        for i in range(0, len(context), inner_batch_size):
            batch_context = context[i:i + inner_batch_size]
            batch_forecast_samples = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                # num_samples=num_samples,
                limit_prediction_length=False
            )
            forecast_samples.append(batch_forecast_samples)
        
        forecast_samples = torch.cat(forecast_samples, dim=0)
        prob_forecast = rearrange(forecast_samples, '(b k) s l -> b s l k', b=B, k=K)
        # prob_forecast = torch.tensor.mean(prob_forecast, dim=1)
        mid = int(prob_forecast.shape[1] / 2)
        return prob_forecast[:, 5, :, :].to(device)