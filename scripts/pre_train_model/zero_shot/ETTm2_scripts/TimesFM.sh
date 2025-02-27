python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 192, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 336, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 720, "seq_len": 96, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 192, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 336, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 720, "seq_len": 320, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 96, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 192, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 336, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 720, "seq_len": 512, "input_patch_len": 32, "output_patch_len": 128, "target_dim": 7, "is_train": 0, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm2/TimesFM"
