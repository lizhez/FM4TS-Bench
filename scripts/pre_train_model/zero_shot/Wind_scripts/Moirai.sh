python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 96, "seq_len": 512, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 192, "seq_len": 512, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 336, "seq_len": 512, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 720, "seq_len": 512, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 96, "seq_len": 96, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 192, "seq_len": 96, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 336, "seq_len": 96, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 720, "seq_len": 96, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 96, "seq_len": 336, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 192, "seq_len": 336, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 336, "seq_len": 336, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moirai" --model-hyper-params '{"horizon": 720, "seq_len": 336, "target_dim": 7, "dataset": "Wind", "freq": "min", "patch_size": 32}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "MOIRAI/ZERO/Wind"

