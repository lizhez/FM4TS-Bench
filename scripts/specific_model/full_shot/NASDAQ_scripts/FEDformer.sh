python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "lr": 0.001, "moving_avg": 25, "horizon": 24, "seq_len": 36, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon": 36, "seq_len": 36, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon": 48, "seq_len": 36, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon": 60, "seq_len": 36, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/FEDformer"



