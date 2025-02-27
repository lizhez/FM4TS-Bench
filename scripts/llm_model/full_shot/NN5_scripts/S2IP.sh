python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.S2IPLLMModel" --model-hyper-params '{"horizon": 24, "seq_len": 36, "label_len": 18, "is_train": 1, "patch_size": 16, "pretrained": 1, "stride": 8, "gpt_layers": 6, "prompt_length": 4, "trend_length": 96, "seasonal_length": 96, "dataset": "NN5", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/S2IPLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.S2IPLLMModel" --model-hyper-params '{"horizon": 36, "seq_len": 36, "label_len": 18, "is_train": 1, "patch_size": 16, "pretrained": 1, "stride": 8, "gpt_layers": 6, "prompt_length": 4, "trend_length": 96, "seasonal_length": 96, "dataset": "NN5", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/S2IPLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.S2IPLLMModel" --model-hyper-params '{"horizon": 48, "seq_len": 36, "label_len": 18, "is_train": 1, "patch_size": 16, "pretrained": 1, "stride": 8, "gpt_layers": 6, "prompt_length": 4, "trend_length": 96, "seasonal_length": 96, "dataset": "NN5", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/S2IPLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.S2IPLLMModel" --model-hyper-params '{"horizon": 60, "seq_len": 36, "label_len": 18, "is_train": 1, "patch_size": 16, "pretrained": 1, "stride": 8, "gpt_layers": 6, "prompt_length": 4, "trend_length": 96, "seasonal_length": 96, "dataset": "NN5", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/S2IPLLM"
