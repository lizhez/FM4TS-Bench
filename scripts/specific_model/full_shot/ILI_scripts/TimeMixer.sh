python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.TimeMixer" --model-hyper-params '{"batch_size":16, "d_ff": 32, "d_model": 16, "e_layers": 2, "lr": 0.01,  "num_epochs": 20, "horizon": 60, "seq_len": 104,"down_sampling_layer": 3,"down_sampling_window": 2,"patience":10, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/TimeMixer"



