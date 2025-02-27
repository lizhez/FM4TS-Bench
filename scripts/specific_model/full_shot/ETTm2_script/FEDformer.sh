python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 96, "seq_len": 96, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 192, "seq_len": 96, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 336, "seq_len": 96, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/FEDformer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 720, "seq_len": 96, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/FEDformer"

