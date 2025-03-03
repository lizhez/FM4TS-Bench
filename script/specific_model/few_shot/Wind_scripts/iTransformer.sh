python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 2, "horizon": 96, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 2, "horizon": 192, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 2, "horizon": 336, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/iTransformer" --adapter "transformer_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 2, "horizon": 720, "norm": true, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/iTransformer" --adapter "transformer_adapter"



