python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 24, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 36, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 48, "seq_len": 104, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/TimesNet"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 60, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ILI/TimesNet"



