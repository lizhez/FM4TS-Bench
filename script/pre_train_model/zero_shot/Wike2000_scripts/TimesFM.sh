python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 24, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 36, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 48, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 60, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 24, "seq_len": 32, "label_len": 16, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 36, "seq_len": 32, "label_len": 16, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 48, "seq_len": 32, "label_len": 16, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 60, "seq_len": 32, "label_len": 16, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 2000, "is_train": 0, "dataset": "Wike2000", "freq": "d"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Wike2000/TimesFM"

