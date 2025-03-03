python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "horizon": 24, "seq_len": 104, "d_ff": 2048, "d_model": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "lr": 0.005, "horizon": 36, "seq_len": 104, "d_ff": 2048, "d_model": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "horizon": 48, "seq_len": 104, "d_ff": 2048, "d_model": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/iTransformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "horizon": 60, "seq_len": 36, "d_ff": 2048, "d_model": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NASDAQ/iTransformer"



