python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":24}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 24, "seq_len": 104, "patience": 20, "lr": 0.005, "sampling_rate": 1}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":36}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 36, "seq_len": 104, "patience": 20, "lr": 0.005, "sampling_rate": 1}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":48}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 48, "seq_len": 104, "patience": 20, "lr": 0.005, "sampling_rate": 1}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/FITS"

python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":60}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 60, "seq_len": 104, "patience": 20, "lr": 0.005, "sampling_rate": 1}' --adapter "fits_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "FULL/NN5/FITS"



