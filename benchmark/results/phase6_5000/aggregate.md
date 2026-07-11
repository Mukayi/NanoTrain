# Phase 6 Benchmark Aggregate

- ddp_2gpu: mode=ddp2, world_size=2, mean_tokens/s=66223.09125288266, mean_step_ms=6.064281673851854, peak_gpu_mb=25.31689453125, final_loss=2.0289430618286133, final_val_loss=2.1021504402160645
- single_cpu: mode=single, world_size=1, mean_tokens/s=11265.38156743592, mean_step_ms=16.454925517997665, peak_gpu_mb=None, final_loss=2.160066604614258, final_val_loss=2.2151308059692383
- single_gpu_activation_checkpointing: mode=single, world_size=1, mean_tokens/s=24675.35981899673, mean_step_ms=7.196023134525888, peak_gpu_mb=23.029296875, final_loss=2.1384153366088867, final_val_loss=2.219453811645508
- single_gpu_bf16: mode=single, world_size=1, mean_tokens/s=37985.3710793133, mean_step_ms=5.466937541006084, peak_gpu_mb=23.21875, final_loss=2.1384153366088867, final_val_loss=2.219453811645508
- tp_2gpu: mode=tp2, world_size=2, mean_tokens/s=26642.155591595652, mean_step_ms=8.152784469849122, peak_gpu_mb=20.06884765625, final_loss=2.1534423828125, final_val_loss=2.202462911605835
- zero1_2gpu: mode=zero1, world_size=2, mean_tokens/s=60907.13423120711, mean_step_ms=6.0939186799502325, peak_gpu_mb=23.5302734375, final_loss=2.02724552154541, final_val_loss=2.102545738220215
- zero2_2gpu: mode=zero2, world_size=2, mean_tokens/s=58875.06474852628, mean_step_ms=6.042985495680081, peak_gpu_mb=21.43212890625, final_loss=1.9440364837646484, final_val_loss=2.0913355350494385
