python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.2, "e_layers": 3, "n_headers": 16, "num_epochs": 100, "patience": 10, "horizon": 96, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Solar/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.2, "e_layers": 3, "n_headers": 16, "num_epochs": 100, "patience": 10, "horizon": 192, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Solar/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.2, "e_layers": 3, "n_headers": 16, "num_epochs": 100, "patience": 10, "horizon": 336, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Solar/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.2, "e_layers": 3, "n_headers": 16, "num_epochs": 100, "patience": 10, "horizon": 720, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Solar/PatchTST"



