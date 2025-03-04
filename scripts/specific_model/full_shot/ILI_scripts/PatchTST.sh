python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 32, "d_model": 512, "e_layers": 4, "factor": 3, "n_headers": 4, "horizon": 24, "seq_len": 104, "d_ff": 2048, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 32, "d_model": 512, "e_layers": 4, "factor": 3, "n_headers": 4, "horizon": 36, "seq_len": 104, "d_ff": 2048, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 32, "d_model": 512, "e_layers": 4, "factor": 3, "n_headers": 4, "horizon": 48, "seq_len": 104, "d_ff": 2048, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 32, "d_model": 512, "e_layers": 4, "factor": 3, "n_headers": 16, "horizon": 60, "seq_len": 104, "d_ff": 2048, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/PatchTST"



