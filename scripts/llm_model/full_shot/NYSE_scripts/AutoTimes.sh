python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"setting": "full", "horizon": 24, "seq_len": 96, "dataset": "NYSE", "freq": "d", "lr": 0.0005, "sampling_rate": 1}' --adapter "autotimes_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NYSE/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"setting": "full", "is_train": 0, "get_train": 1, "horizon": 36, "seq_len": 96, "label_len": 18, "dataset": "NYSE", "freq": "d", "lr": 0.0005, "sampling_rate": 1}' --adapter "autotimes_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NYSE/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"setting": "full", "is_train": 0, "get_train": 1, "horizon": 48, "seq_len": 96, "label_len": 18, "dataset": "NYSE", "freq": "d", "lr": 0.0005, "sampling_rate": 1}' --adapter "autotimes_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NYSE/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"setting": "full", "is_train": 0, "get_train": 1, "horizon": 60, "seq_len": 96, "label_len": 18, "dataset": "NYSE", "freq": "d", "lr": 0.0005, "sampling_rate": 1}' --adapter "autotimes_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NYSE/AutoTime"



