# NanoTrain Benchmark Aggregate

- activation_off: mode=single, world_size=1, tokens/s=61271.32998239016, speedup=0.7410012382474563, efficiency=74.10012382474564%, peak_gpu_mb=1865.57666015625, step_peak_gpu_mb=1865.57666015625, final_val_loss=2.4821081161499023
- activation_on: mode=single, world_size=1, tokens/s=49927.61048660706, speedup=0.6038129285580314, efficiency=60.38129285580314%, peak_gpu_mb=1410.26611328125, step_peak_gpu_mb=1410.26611328125, final_val_loss=2.481266498565674
- ddp_2gpu_memory: mode=ddp2, world_size=2, tokens/s=82687.21672760375, speedup=1.0, efficiency=50.0%, peak_gpu_mb=2182.64404296875, step_peak_gpu_mb=2182.64404296875, final_val_loss=2.408958673477173
- zero1_2gpu_memory: mode=zero1, world_size=2, tokens/s=65164.40700891173, speedup=0.7880832078745937, efficiency=39.40416039372969%, peak_gpu_mb=1796.8056640625, step_peak_gpu_mb=1796.8056640625, final_val_loss=2.3801822662353516
- zero2_2gpu_memory: mode=zero2, world_size=2, tokens/s=61228.66731994181, speedup=0.7404852859136283, efficiency=37.02426429568141%, peak_gpu_mb=1482.98828125, step_peak_gpu_mb=1482.98828125, final_val_loss=2.4533705711364746
