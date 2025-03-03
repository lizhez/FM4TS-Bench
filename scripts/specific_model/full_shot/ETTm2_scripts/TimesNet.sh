python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 96, "seq_len": 96, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 32, "d_model": 32, "factor": 3, "num_epochs": 1, "horizon": 192, "seq_len": 96, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 32, "d_model": 32, "factor": 3, "horizon": 336, "seq_len": 96, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 32, "d_model": 16, "factor": 3, "num_epochs": 1, "horizon": 720, "seq_len": 96, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTm2/TimesNet"



