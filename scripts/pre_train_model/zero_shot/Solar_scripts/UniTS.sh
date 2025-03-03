python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 96, "seq_len": 96, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 192, "seq_len": 96, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 336, "seq_len": 96, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 720, "seq_len": 96, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 96, "seq_len": 336, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 192, "seq_len": 336, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 336, "seq_len": 336, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 720, "seq_len": 336, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 96, "seq_len": 512, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 192, "seq_len": 512, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 336, "seq_len": 512, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.UniTS" --model-hyper-params '{"horizon": 720, "seq_len": 512, "target_dim": 137, "is_train": 0, "dataset": "Solar", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZERO/Solar/UniTS"

