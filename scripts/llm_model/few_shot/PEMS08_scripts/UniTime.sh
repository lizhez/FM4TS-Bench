python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 96, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 96, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 96, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 96, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 336, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 336, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 336, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 336, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 32, "max_backcast_len": 336, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 96, "seq_len": 512, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 60, "max_backcast_len": 512, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 192, "seq_len": 512, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 60, "max_backcast_len": 512, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 336, "seq_len": 512, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 60, "max_backcast_len": 512, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "LLM.UniTimeModel" --model-hyper-params '{"horizon": 720, "seq_len": 512, "is_train": 1, "enc_in": 170, "stride": 16, "max_token_num": 60, "max_backcast_len": 512, "freq": "min", "dataset": "PEMS08", "sampling_rate": 0.05}' --adapter "llm_adapter"  --gpus 2  --num-workers 1  --timeout 60000  --save-path "FEW/PEMS08/UniTime"



