# Fast nanoGPT baseline smoke config.
#
# Run from the nanoGPT directory:
#   python train.py ../configs/nanogpt_shakespeare_char_smoke.py
#
# This config intentionally keeps nanoGPT's original source untouched while
# giving NanoTrain a quick regression target for Phase 1 migration work.

out_dir = "../out/nanogpt-shakespeare-char-smoke"
eval_interval = 10
eval_iters = 5
log_interval = 1
always_save_checkpoint = False

wandb_log = False

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 4
block_size = 32

n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 20
lr_decay_iters = 20
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 2

compile = False
device = "cpu"
dtype = "float32"
