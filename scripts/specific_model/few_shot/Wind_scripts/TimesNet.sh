python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 64, "d_model": 64, "factor": 3, "horizon": 96, "norm": true, "seq_len": 96, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 128, "d_model": 64, "horizon": 192, "norm": true, "seq_len": 96, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 32, "d_model": 32, "factor": 3, "horizon": 336, "norm": true, "num_epochs": 1, "seq_len": 512, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 64, "d_model": 32, "factor": 3, "horizon": 720, "norm": true, "seq_len": 336, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/TimesNet" --adapter "transformer_adapter"



