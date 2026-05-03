import torch

out_dir = "out-uncertainty-byte"
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "nanogpt-uncertainty"
wandb_run_name = "byte-shakespeare"

dataset = "uncertainty_byte"

gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1
bias = False

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
dtype = "float32"
compile = False
