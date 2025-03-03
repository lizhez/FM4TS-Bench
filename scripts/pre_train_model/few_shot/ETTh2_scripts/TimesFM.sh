python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "get_train": 1, "sampling_rate": 0.05, "dataset": "ETTh2", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh2/TimesFM"

