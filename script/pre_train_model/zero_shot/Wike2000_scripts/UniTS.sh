python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 24, "seq_len": 36, "label_len": 18, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 36, "seq_len": 36, "label_len": 18, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 48, "seq_len": 36, "label_len": 18, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 60, "seq_len": 36, "label_len": 18, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 24, "seq_len": 104, "label_len": 52, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 36, "seq_len": 104, "label_len": 52, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 48, "seq_len": 104, "label_len": 52, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 60, "seq_len": 104, "label_len": 52, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/UniTS"

