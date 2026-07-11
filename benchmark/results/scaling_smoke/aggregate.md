# NanoTrain Benchmark Aggregate

- ddp_2gpu: mode=ddp2, world_size=2, tokens/s=29299.558173624227, speedup=1.956422118094491, efficiency=97.82110590472455%, peak_gpu_mb=476.4921875, step_peak_gpu_mb=476.4921875, final_val_loss=3.827190399169922
- ddp_4gpu: mode=ddp4, world_size=4, tokens/s=56471.210093894784, speedup=3.770757354379264, efficiency=94.26893385948159%, peak_gpu_mb=476.4921875, step_peak_gpu_mb=476.4921875, final_val_loss=3.8269920349121094
- ddp_8gpu: mode=ddp8, world_size=8, tokens/s=96119.49111111733, speedup=6.418195703683396, efficiency=80.22744629604244%, peak_gpu_mb=476.4921875, step_peak_gpu_mb=476.4921875, final_val_loss=3.8259849548339844
- single_gpu_bf16: mode=single, world_size=1, tokens/s=14976.092277141759, speedup=1.0, efficiency=100.0%, peak_gpu_mb=435.5029296875, step_peak_gpu_mb=435.5029296875, final_val_loss=3.8275413513183594
