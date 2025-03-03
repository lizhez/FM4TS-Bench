python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE",  "horizon": 96, "seq_len": 336, "patience": 3, "lr": 0.0005, "sampling_rate": 0.05}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh1/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE",  "horizon": 192, "seq_len": 512, "patience": 3, "lr": 0.0005, "sampling_rate": 0.05}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh1/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE",  "horizon": 336, "seq_len": 512, "patience": 3, "lr": 0.0005, "sampling_rate": 0.05}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh1/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE",  "horizon": 720, "seq_len": 512, "patience": 3, "lr": 0.0005, "sampling_rate": 0.05}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTh1/FITS"



