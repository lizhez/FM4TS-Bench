python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "Solar", "freq": "min", "target_dim": 137, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "TTM/ZERO/Solar"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 192, "seq_len": 512, "dataset": "Solar", "freq": "min", "target_dim": 137, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "TTM/ZERO/Solar"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 336, "seq_len": 512, "dataset": "Solar", "freq": "min", "target_dim": 137, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "TTM/ZERO/Solar"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 720, "seq_len": 512, "dataset": "Solar", "freq": "min", "target_dim": 137, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "TTM/ZERO/Solar"

