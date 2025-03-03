python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "Weather", "freq": "min", "target_dim": 21, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "Weather", "freq": "min", "target_dim": 21, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "Weather", "freq": "min", "target_dim": 21, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "Weather", "freq": "min", "target_dim": 21, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Weather/TTM/FULL"
