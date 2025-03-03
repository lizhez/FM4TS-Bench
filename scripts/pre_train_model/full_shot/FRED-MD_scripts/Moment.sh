python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 24, "seq_len": 36, "label_len": 18, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 36, "seq_len": 36, "label_len": 18, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 48, "seq_len": 36, "label_len": 18, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 60, "seq_len": 36, "label_len": 18, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 24, "seq_len": 104, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 36, "seq_len": 104, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 48, "seq_len": 104, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "pre_train.Moment" --model-hyper-params '{"horizon": 60, "seq_len": 104, "target_dim": 107, "is_train": 1, "sampling_rate": 1, "dataset": "FRED-MD", "freq": "m"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/FRED-MD/Moment"

