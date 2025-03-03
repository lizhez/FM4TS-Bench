python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 192, "seq_len": 96, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 336, "seq_len": 96, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 720, "seq_len": 96, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"horizon": 96, "seq_len": 384, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 192, "seq_len": 384, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 336, "seq_len": 384, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 720, "seq_len": 384, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"horizon": 96, "seq_len": 672, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 192, "seq_len": 672, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 336, "seq_len": 672, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"

python ./scripts/run.py --config-path "autotimes_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.AutoTimesModel" --model-hyper-params '{"is_train": 0, "get_train": 1, "horizon": 720, "seq_len": 672, "dataset": "Wind", "freq": "min", "lr": 0.0005, "sampling_rate": 0.05}' --adapter "autotimes_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Few/Wind/AutoTime"



