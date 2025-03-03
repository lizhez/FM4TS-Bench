python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 24, "seq_len": 96, "target_dim": 5, "is_train": 0, "dataset": "NYSE", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/NYSE/Timer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 36, "seq_len": 96, "target_dim": 5, "is_train": 0, "dataset": "NYSE", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/NYSE/Timer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 48, "seq_len": 96, "target_dim": 5, "is_train": 0, "dataset": "NYSE", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/NYSE/Timer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 60, "seq_len": 96, "target_dim": 5, "is_train": 0, "dataset": "NYSE", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/NYSE/Timer"

