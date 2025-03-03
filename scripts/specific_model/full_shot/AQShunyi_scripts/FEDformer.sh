python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "horizon": 96, "moving_avg": 25, "norm": true, "seq_len": 512, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "AQShunyi/FEDformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "horizon": 192, "moving_avg": 25, "norm": true, "seq_len": 336, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "AQShunyi/FEDformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "dropout": 0.05, "factor": 3, "horizon": 336, "moving_avg": 25, "norm": true, "seq_len": 96, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "AQShunyi/FEDformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 720, "moving_avg": 25, "norm": true, "seq_len": 512, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "AQShunyi/FEDformer" --adapter "transformer_adapter"

