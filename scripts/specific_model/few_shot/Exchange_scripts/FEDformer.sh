python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "moving_avg": 25, "horizon": 96, "seq_len": 96, "sampling_rate": 0.05}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/Exchange/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 192, "seq_len": 96, "sampling_rate": 0.05}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/Exchange/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 336, "seq_len": 96, "sampling_rate": 0.05}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/Exchange/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"factor": 3, "horizon": 720, "seq_len": 96, "d_ff": 2048, "d_model": 512, "sampling_rate": 0.05}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/Exchange/FEDformer"



