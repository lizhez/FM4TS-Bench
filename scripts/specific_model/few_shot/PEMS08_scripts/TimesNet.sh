python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 64, "d_ff": 1024, "d_model": 512, "factor": 3, "horizon": 96, "norm": true, "seq_len": 336, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "factor": 3, "horizon": 192, "norm": true, "seq_len": 336, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 336, "norm": true, "seq_len": 336, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/TimesNet" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "factor": 3, "horizon": 720, "norm": true, "seq_len": 512, "top_k": 5, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/TimesNet" --adapter "transformer_adapter"



