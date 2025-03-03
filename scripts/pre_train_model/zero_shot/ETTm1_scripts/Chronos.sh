python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 512, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 512, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 720, "seq_len": 512, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 96, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 96, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 96, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 720, "seq_len": 96, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 336, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 336, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 336, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 720, "seq_len": 336, "dataset": "ETTm1", "freq": "min", "target_dim": 7, "is_train": 0}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/ETTm1/Chronos"

