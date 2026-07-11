# Phase 6 Benchmark Aggregate

- ddp_2gpu: mode=ddp2, world_size=2, mean_tokens/s=60146.25838713696, mean_step_ms=5.116324675710578, peak_gpu_mb=25.31689453125, final_loss=3.5452880859375, final_val_loss=3.4260621070861816
- single_cpu: mode=single, world_size=1, mean_tokens/s=9083.765585481198, mean_step_ms=20.694230732164886, peak_gpu_mb=None, final_loss=3.3347599506378174, final_val_loss=3.1503548622131348
- single_gpu_activation_checkpointing: mode=single, world_size=1, mean_tokens/s=21410.74080391534, mean_step_ms=6.778553912514134, peak_gpu_mb=23.029296875, final_loss=3.619140625, final_val_loss=3.487060546875
- single_gpu_bf16: mode=single, world_size=1, mean_tokens/s=34048.95210302171, mean_step_ms=4.596697656731856, peak_gpu_mb=23.21875, final_loss=3.619140625, final_val_loss=3.487060546875
- tp_2gpu: mode=tp2, world_size=2, mean_tokens/s=19868.409953597424, mean_step_ms=7.469076859323602, peak_gpu_mb=20.06884765625, final_loss=3.5380859375, final_val_loss=3.5877928733825684
- zero1_2gpu: mode=zero1, world_size=2, mean_tokens/s=50413.23632377299, mean_step_ms=5.650583066438374, peak_gpu_mb=23.5302734375, final_loss=3.5506591796875, final_val_loss=3.433666944503784
- zero2_2gpu: mode=zero2, world_size=2, mean_tokens/s=47630.56031373998, mean_step_ms=5.908012390136719, peak_gpu_mb=21.43212890625, final_loss=3.6002197265625, final_val_loss=3.4895997047424316
