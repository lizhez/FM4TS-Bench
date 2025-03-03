python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 336, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 512, "dataset": "PEMS08", "freq": "min", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/LLMMixer"



