python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 96, "seq_len": 336, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 192, "seq_len": 336, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 336, "seq_len": 336, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 720, "seq_len": 336, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 96, "seq_len": 512, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 192, "seq_len": 512, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 336, "seq_len": 512, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "LLM.TimeLLMsModel" --model-hyper-params '{"horizon": 720, "seq_len": 512, "is_train": 1, "enc_in": 7, "d_model": 32, "dataset": "Wind", "freq": "min", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "FEW/Wind/TimeLLM"



