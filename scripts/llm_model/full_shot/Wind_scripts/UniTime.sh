python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 50, "max_backcast_len": 96, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 50, "max_backcast_len": 96, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 50, "max_backcast_len": 96, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 50, "max_backcast_len": 96, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 336, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 336, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 336, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 336, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 512, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 43, "max_backcast_len": 512, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 512, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 43, "max_backcast_len": 512, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 512, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 43, "max_backcast_len": 512, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 512, "is_train": 1, "enc_in": 7, "stride": 16, "max_token_num": 43, "max_backcast_len": 512, "freq": "min", "dataset": "Wind", "sampling_rate": 1}' --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/Wind/UniTime"
