python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 2048, "d_model": 512, "horizon": 96, "lr": 0.001, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/DLinear" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "d_ff": 2048, "d_model": 512, "horizon": 192, "lr": 0.005, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/DLinear" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 2048, "d_model": 512, "horizon": 336, "lr": 0.005, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/DLinear" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 32, "d_ff": 2048, "d_model": 512, "horizon": 720, "lr": 0.005, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/DLinear" --adapter "transformer_adapter"



