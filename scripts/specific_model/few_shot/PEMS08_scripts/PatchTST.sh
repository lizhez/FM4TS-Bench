python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 32, "d_ff": 2048, "d_model": 512, "horizon": 96, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/PatchTST" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 3, "horizon": 192, "n_headers": 16, "norm": true, "num_epochs": 100, "patience": 10, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/PatchTST" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 3, "horizon": 336, "n_headers": 16, "norm": true, "num_epochs": 100, "patience": 10, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/PatchTST" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "dropout": 0.1, "e_layers": 3, "horizon": 720, "n_headers": 16, "norm": true, "num_epochs": 100, "patience": 10, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/PEMS08/PatchTST" --adapter "transformer_adapter"



