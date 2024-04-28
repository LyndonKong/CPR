import torch
import wandb
import gym
import numpy as np
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from torch.utils.data import DataLoader

from src.utils.buffer import TransitionSimpleReplay
from offlinerllib.module.actor import SquashedDeterministicActor
from offlinerllib.module.critic import Critic
from offlinerllib.module.net.mlp import MLP
from src.policy import CPRPolicy
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_offline_policy

stage = "online"
args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(
    log_path=f"./log/cpr/{stage}/{args.name}",
    name=exp_name, loggers_config={
        "FileLogger": {"activate": not args.debug}, 
        "TensorboardLogger": {"activate": not args.debug}, 
        "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    }
)
setup(args, logger)

# do not need the datasets, just init the normalized env wrapper
if args.normalize_obs:
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    eval_env, _ = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    del dataset
else:
    env = gym.make(args.task)
    eval_env = gym.make(args.task)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

online_buffer = TransitionSimpleReplay(
    max_size=args.max_buffer_size, 
    field_specs={
        "observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "actions": {"shape": [action_shape, ], "dtype": np.float32}, 
        "next_observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "rewards": {"shape": [1, ], "dtype": np.float32, }, 
        "terminals": {"shape": [1, ], "dtype": np.float32, }, 
    }
)
online_buffer.reset()

actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims)
actor = SquashedDeterministicActor(
    backend=actor_backend,
    input_dim=args.hidden_dims[-1],
    output_dim=action_shape,
    device=args.device
).to(args.device)
actor_optim = torch.optim.Adam(
    actor.parameters(),
    lr=args.actor_lr,
)

critic = Critic(
    backend=torch.nn.Identity(),
    input_dim=obs_shape + action_shape,
    hidden_dims=args.hidden_dims,
    ensemble_size=2,
    device=args.device
).to(args.device)
critic_optim = torch.optim.Adam(
    critic.parameters(),
    lr=args.critic_lr,
)

policy = CPRPolicy(
    actor=actor, 
    critic=critic, 
    actor_optim=actor_optim, 
    critic_optim=critic_optim,
    alpha=args.alpha, 
    actor_update_interval=args.actor_update_interval, 
    policy_noise=args.policy_noise, 
    noise_clip=args.noise_clip, 
    tau=args.tau, 
    discount=args.discount, 
    max_action=args.max_action, 
    device=args.device,
    eta=args.eta
).to(args.device)

# load the offline pretrained dataset
model_path = f"./out/td3bc/offline/corl/{args.task}/seed{args.seed}/policy/policy_1000.pt"
policy.load_state_dict(torch.load(model_path))


# main loop
obs, terminal = env.reset(), False
cur_traj_length = cur_traj_return = 0
all_traj_lengths = [0]
all_traj_returns = [0]

for i_epoch in trange(1, args.num_epoch+1):
    actor_train_metrics = {}
    critic_train_metrics = {}
    if i_epoch == 1:
        eval_metrics = eval_offline_policy(eval_env, policy, args.eval_episode, seed=args.seed)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
    for i_step in range(args.step_per_epoch):
        action = policy.select_action(obs, deterministic=False)
        next_obs, reward, terminal, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward
        if cur_traj_length >= args.max_trajectory_length:
            terminal = False
        online_buffer.add_sample({
            "observations": obs,
            "actions": action, 
            "next_observations": next_obs, 
            "rewards": reward, 
            "terminals": terminal, 
        })
        obs = next_obs
        if terminal or cur_traj_length >= args.max_trajectory_length:
            obs = env.reset()
            all_traj_returns.append(cur_traj_return)
            all_traj_lengths.append(cur_traj_length)
            cur_traj_length = cur_traj_return = 0
        batch_data = online_buffer.random_batch(args.batch_size)
        if i_epoch > args.revitalize_interval:
            train_metrics = policy.update(batch_data, stage="finetune")
        else:
            train_metrics = policy.update(batch_data, stage="alignment")
    
    if i_epoch % args.revitalize_interval == 0:
        policy.actor_revitalize()
        loader = DataLoader(online_buffer, batch_size=args.batch_size, shuffle=True, pin_memory=False)
        # the fitting round is fixed
        for _ in range(32):
            for _, batch_data in enumerate(loader):
                actor_train_metrics = policy.update(batch_data, stage="revitalization")
        del loader

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_offline_policy(eval_env, policy, args.eval_episode, seed=args.seed)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
    
    if i_epoch % args.log_interval == 0:
        if len(actor_train_metrics) > 0:
            logger.log_scalars("", actor_train_metrics, step=i_epoch)
        if len(critic_train_metrics) > 0:
            logger.log_scalars("", critic_train_metrics, step=i_epoch)
        logger.log_scalars("rollout", {
            "episode_return": all_traj_returns[-1], 
            "episode_length": all_traj_lengths[-1]
        }, step=i_epoch)
    
    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/cpr/{stage}/{args.name}/{args.task}/seed{args.seed}/policy/")
