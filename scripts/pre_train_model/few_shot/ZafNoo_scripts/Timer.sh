python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 96, "seq_len": 384, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 192, "seq_len": 384, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 336, "seq_len": 384, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 720, "seq_len": 384, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 96, "seq_len": 672, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 192, "seq_len": 672, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 336, "seq_len": 672, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 720, "seq_len": 672, "target_dim": 11, "is_train": 1, "sampling_rate": 0.05, "dataset": "ZafNoo", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ZafNoo/TimerModel"

