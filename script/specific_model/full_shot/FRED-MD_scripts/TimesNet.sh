python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 256, "d_model": 128, "factor": 3, "horizon": 24, "norm": true, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 36, "norm": true, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 256, "d_model": 128, "factor": 3, "horizon": 48, "norm": true, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 60, "norm": true, "seq_len": 36, "top_k": 5, "sampling_rate": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/TimesNet" --adapter "transformer_adapter"

