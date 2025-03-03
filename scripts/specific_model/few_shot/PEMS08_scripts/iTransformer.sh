python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 512, "e_layers": 3, "horizon": 96, "lr": 0.0005, "norm": true, "seq_len": 336, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 512, "e_layers": 3, "horizon": 192, "lr": 0.0005, "norm": true, "seq_len": 336, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 512, "e_layers": 3, "horizon": 336, "lr": 0.0005, "norm": true, "seq_len": 336, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 512, "e_layers": 3, "horizon": 720, "lr": 0.0005, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/iTransformer" --adapter "transformer_adapter"



