python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 24, "seq_len": 104, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 36, "seq_len": 104, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 48, "seq_len": 104, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 60, "seq_len": 104, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 24, "seq_len": 36, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 36, "seq_len": 36, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 48, "seq_len": 36, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.LLMMixerModel" --model-hyper-params '{"horizon": 60, "seq_len": 36, "dataset": "NN5", "lr": 0.001, "use_norm": 1, "d_model": 16, "n_heads": 4, "is_train": 1, "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/LLMMixer"


