# Fill-in lab copy of the DOWSING training config.
# The complete reference lives at config/train_uncertainty_byte.py.

out_dir = "out-uncertainty-byte"
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
dataset = "uncertainty_byte"

gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# TODO[DOWSING-01]: Choose a small model that still has several layers to probe.
# Keeping 6 layers makes the later layerwise safety analysis interesting, while
# smaller width/head counts keep training realistic for a lab run.
n_layer = None
n_head = None
n_embd = None

dropout = 0.1
bias = False

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = "cuda"
compile = False
