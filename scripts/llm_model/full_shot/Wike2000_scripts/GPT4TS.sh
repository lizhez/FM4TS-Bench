python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 24, "seq_len": 36, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 36, "seq_len": 36, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 48, "seq_len": 36, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 60, "seq_len": 36, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 24, "seq_len": 104, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 36, "seq_len": 104, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 48, "seq_len": 104, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.GPT4TSModel" --model-hyper-params '{"horizon": 60, "seq_len": 104, "is_train": 1, "is_gpt": 1, "patch_size": 16, "pretrain": 1, "stride": 8, "gpt_layers": 3, "d_model": 768, "freeze": 1, "lr": 0.0001, "freq": "d", "dataset": "Wike2000", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/GPT4TS"
