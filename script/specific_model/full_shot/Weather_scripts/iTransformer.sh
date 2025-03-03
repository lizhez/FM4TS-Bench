python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 2, "horizon": 96, "seq_len": 336, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Weather/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 2, "horizon": 192, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Weather/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 1, "horizon": 336, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Weather/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 1, "horizon": 720, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Weather/iTransformer"



