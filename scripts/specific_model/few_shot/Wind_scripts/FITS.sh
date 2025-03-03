python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 14, "base_T": 96, "horizon": 96, "loss": "MSE", "lr": 0.0005, "norm": true, "patience": 20, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/FITS" --adapter "fits_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 14, "base_T": 96, "horizon": 192, "loss": "MSE", "lr": 0.0005, "norm": true, "patience": 20, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/FITS" --adapter "fits_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 14, "base_T": 96, "horizon": 336, "loss": "MSE", "lr": 0.0005, "norm": true, "patience": 20, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/FITS" --adapter "fits_adapter"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "Wind.csv" --strategy-args '{"horizon": 720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 14, "base_T": 96, "horizon": 720, "loss": "MSE", "lr": 0.0005, "norm": true, "patience": 20, "seq_len": 512, "sampling_rate": 0.05}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FEW/sp/Wind/FITS" --adapter "fits_adapter"



