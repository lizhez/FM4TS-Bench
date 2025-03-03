python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"batch_size": 1024, "horizon": 720, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 720, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 720, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 192, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 336, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "pre_train.Chronos" --model-hyper-params '{"batch_size": 512, "horizon": 720, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "target_dim": 170, "is_train": 1, "sampling_rate": 0.05}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/Chronos"

