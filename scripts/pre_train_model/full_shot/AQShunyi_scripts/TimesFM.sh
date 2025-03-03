python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "AQShunyi.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 11, "is_train": 0, "get_train": 1, "sampling_rate": 1, "dataset": "AQShunyi", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/AQShunyi/TimesFM"

