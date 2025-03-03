python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 24, "seq_len": 36, "label_len": 18, "dataset": "NN5", "freq": "d", "target_dim": 111, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 36, "seq_len": 36, "label_len": 18, "dataset": "NN5", "freq": "d", "target_dim": 111, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 48, "seq_len": 36, "label_len": 18, "dataset": "NN5", "freq": "d", "target_dim": 111, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TTM/FULL"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 60, "seq_len": 36, "label_len": 18, "dataset": "NN5", "freq": "d", "target_dim": 111, "is_train": 1, "sampling_rate": 1}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TTM/FULL"
