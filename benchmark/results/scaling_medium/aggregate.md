# NanoTrain Benchmark Aggregate

- ddp_2gpu: mode=ddp2, world_size=2, tokens/s=609323.7144610734, speedup=1.6201559653336337, efficiency=81.00779826668169%, peak_gpu_mb=557.3544921875, step_peak_gpu_mb=557.3544921875, final_val_loss=1.5859553813934326
- ddp_4gpu: mode=ddp4, world_size=4, tokens/s=1124311.387757104, speedup=2.9894779384687453, efficiency=74.73694846171864%, peak_gpu_mb=556.3544921875, step_peak_gpu_mb=556.3544921875, final_val_loss=1.523510456085205
- ddp_8gpu: mode=ddp8, world_size=8, tokens/s=2143187.0648919935, speedup=5.698608515642259, efficiency=71.23260644552823%, peak_gpu_mb=560.1044921875, step_peak_gpu_mb=560.1044921875, final_val_loss=1.5592666864395142
- single_gpu_bf16: mode=single, world_size=1, tokens/s=376089.5416853261, speedup=1.0, efficiency=100.0%, peak_gpu_mb=517.4677734375, step_peak_gpu_mb=517.4677734375, final_val_loss=1.736901879310608
