python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 16, "d_ff": 1024, "d_model": 512, "lr": 0.001, "horizon": 96, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 16, "d_ff": 1024, "d_model": 512, "lr": 0.0005, "num_epochs": 20, "horizon": 192, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 16, "d_ff": 1024, "d_model": 512, "lr": 0.0005, "num_epochs": 20, "horizon": 336, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/PatchTST"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 16, "d_ff": 1024, "d_model": 512, "lr": 0.0005, "num_epochs": 20, "horizon": 720, "seq_len": 512, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/PatchTST"



