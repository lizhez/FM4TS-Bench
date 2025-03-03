python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 512, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 512, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 512, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 512, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":96}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 96, "seq_len": 336, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":192}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 192, "seq_len": 336, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":336}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 336, "seq_len": 336, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon":720}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 720, "seq_len": 336, "dataset": "ETTh2", "freq": "h", "lr": 0.001, "use_norm": 1, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/ETTh2/LLMMixer"



