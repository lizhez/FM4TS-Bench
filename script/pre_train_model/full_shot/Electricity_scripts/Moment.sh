python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 96, "seq_len": 96, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 192, "seq_len": 96, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 336, "seq_len": 96, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 720, "seq_len": 96, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 96, "seq_len": 336, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 192, "seq_len": 336, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 336, "seq_len": 336, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 720, "seq_len": 336, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 96, "seq_len": 512, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 192, "seq_len": 512, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 336, "seq_len": 512, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 720, "seq_len": 512, "target_dim": 321, "is_train": 1, "sampling_rate": 1, "dataset": "Electricity", "freq": "h"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Electricity/Moment"

