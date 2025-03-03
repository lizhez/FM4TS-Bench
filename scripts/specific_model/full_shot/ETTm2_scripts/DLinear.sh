python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "horizon": 96, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/DLinear"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 192, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/DLinear"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "lr": 0.001, "horizon": 336, "seq_len": 336, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/DLinear"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "lr": 0.001, "horizon": 720, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/DLinear"



