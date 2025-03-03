python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"sampling_rate": 0.05, "d_ff": 512, "d_model": 256, "horizon": 96, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/FEDformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"sampling_rate": 0.05, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "horizon": 192, "moving_avg": 25, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/FEDformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"sampling_rate": 0.05, "d_ff": 512, "d_model": 256, "horizon": 336, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/FEDformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"sampling_rate": 0.05, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "horizon": 720, "moving_avg": 25, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/FEDformer" --adapter "transformer_adapter"



