# NanoTrain Benchmark Aggregate

- activation_off: mode=single, world_size=1, tokens/s=7173.252992288068, speedup=0.5300139040246514, efficiency=53.00139040246514%, peak_gpu_mb=1342.376953125, step_peak_gpu_mb=1342.376953125, final_val_loss=4.201048374176025
- activation_on: mode=single, world_size=1, tokens/s=6128.134337717705, speedup=0.45279267414842184, efficiency=45.27926741484218%, peak_gpu_mb=1344.126953125, step_peak_gpu_mb=1344.126953125, final_val_loss=4.201107025146484
- ddp_2gpu_memory: mode=ddp2, world_size=2, tokens/s=13534.084554797704, speedup=1.0, efficiency=50.0%, peak_gpu_mb=1659.501953125, step_peak_gpu_mb=1659.501953125, final_val_loss=4.2020182609558105
- zero1_2gpu_memory: mode=zero1, world_size=2, tokens/s=13166.012717024367, speedup=0.9728040831810187, efficiency=48.64020415905094%, peak_gpu_mb=1520.3876953125, step_peak_gpu_mb=1520.3876953125, final_val_loss=4.2024970054626465
- zero2_2gpu_memory: mode=zero2, world_size=2, tokens/s=7803.31729083839, speedup=0.5765677951282038, efficiency=28.82838975641019%, peak_gpu_mb=1194.7626953125, step_peak_gpu_mb=1194.7626953125, final_val_loss=4.286109924316406
