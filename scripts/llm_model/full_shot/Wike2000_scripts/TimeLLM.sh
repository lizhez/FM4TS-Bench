python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 24, "seq_len": 36, "label_len": 18, "is_train": 1, "enc_in": 2000, "freq": "min", "d_model": 32, "dataset": "Wike2000", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 36, "seq_len": 36, "label_len": 18, "is_train": 1, "enc_in": 2000, "freq": "min", "d_model": 32, "dataset": "Wike2000", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 48, "seq_len": 36, "label_len": 18, "is_train": 1, "enc_in": 2000, "freq": "min", "d_model": 32, "dataset": "Wike2000", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 60, "seq_len": 36, "label_len": 18, "is_train": 1, "enc_in": 2000, "freq": "min", "d_model": 32, "dataset": "Wike2000", "freq": "d", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wike2000/TimeLLM"
