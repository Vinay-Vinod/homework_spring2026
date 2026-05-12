import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def run_offline_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, start_step: int = 0):
    """
    Run offline training loop.
    Returns (agent, env, dataset) so the online loop can reuse them.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env, dataset = config["make_env_and_dataset"]()

    spec_batch = dataset.sample(1)
    agent_class = agents[config["agent"]]
    agent = agent_class(
        spec_batch['observations'].shape[1:],
        spec_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    for step in tqdm.trange(start_step, start_step + config["offline_training_steps"] + 1, dynamic_ncols=True):
        batch = dataset.sample(config["batch_size"])
        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            eval_rollouts = utils.sample_n_trajectories(
                env, agent, args.num_eval_trajectories, ep_len,
            )
            eval_successes = [t["episode_statistics"]["s"] for t in eval_rollouts]
            eval_logger.log(
                {"eval/success_rate": float(np.mean(eval_successes))},
                step=step,
            )

    return agent, env, dataset


def run_online_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, agent, env, dataset, start_step: int = 0):
    """
    Run online training loop (collect data in env, store in replay buffer, update).
    """
    online_buffer = ReplayBuffer(capacity=config['replay_buffer_capacity'])
    if config["offline_data"] > 0:
        b = dataset.sample(min(config["offline_data"], online_buffer.max_size))
        for i in range(len(b["observations"])):
            online_buffer.insert(
                observation=b["observations"][i],
                action=b["actions"][i],
                reward=b["rewards"][i],
                next_observation=b["next_observations"][i],
                done=b["dones"][i],
            )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    obs, i = env.reset()

    for step in tqdm.trange(start_step, start_step + config["online_training_steps"] + 1, dynamic_ncols=True):
        with torch.no_grad():
            action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        online_buffer.insert(
            observation=obs,
            action=action,
            reward=np.float32(reward),
            next_observation=next_obs,
            done=np.float32(terminated),
        )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if (step - start_step >= config["wsrl_steps"] and len(online_buffer) >= config["batch_size"]):
            batch = online_buffer.sample(config["batch_size"])
            batch = {
                k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
            }

            metrics = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
                step,
            )

            if step % args.log_interval == 0:
                train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            eval_rollouts = utils.sample_n_trajectories(
                env, agent, args.num_eval_trajectories, ep_len,
            )
            eval_successes = [t["episode_statistics"]["s"] for t in eval_rollouts]
            eval_logger.log(
                {"eval/success_rate": float(np.mean(eval_successes))},
                step=step,
            )
            obs, i = env.reset()

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    parser.add_argument("--offline_data", type=int, default=0)
    
    # WSRL
    parser.add_argument("--wsrl_steps", type=int, default=0)

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    config['offline_data'] = args.offline_data
    config['wsrl_steps'] = args.wsrl_steps

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.offline_data > 0:
        exp_name = f"{exp_name}_od{args.offline_data}"
    if args.wsrl_steps > 0:
        exp_name = f"{exp_name}_wsrl{args.wsrl_steps}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    agent = None
    env = None
    dataset = None
    start_step = 0

    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        agent, env, dataset = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps

    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        if agent is None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
            env, dataset = config["make_env_and_dataset"]()
            spec_batch = dataset.sample(1)
            agent_class = agents[config["agent"]]
            agent = agent_class(
                spec_batch['observations'].shape[1:],
                spec_batch['actions'].shape[-1],
                **config["agent_kwargs"],
            )
        run_online_training_loop(config, train_logger, eval_logger, args, agent, env, dataset, start_step=start_step)
    else:
        if agent is not None:
            dump_log(agent, train_logger, eval_logger, config, args.save_dir)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
