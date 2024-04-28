from UtilsRL.misc import NameSpace

seed = 0
task = None

discount = 0.99
tau = 0.005

hidden_dims = [256, 256]
actor_lr = 0.0003
critic_lr = 0.0003
batch_size = 256

actor_update_interval = 2
policy_noise = 0.2
noise_clip = 0.5
max_action = 1.0
max_buffer_size = 1000000

num_epoch = 300
step_per_epoch = 1000
warmup_epoch = 10
max_trajectory_length = 1000

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
revitalize_interval = 10
warmup_epoch = 2
eta = 1

name = "original"

policy_noise = 0.2
noise_clip = 0.5
exploration_noise = 0.1

class wandb(NameSpace):
    entity = None
    project = None

debug = False
