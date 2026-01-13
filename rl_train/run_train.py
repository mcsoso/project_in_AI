import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
import time
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
import matplotlib
import multiprocessing as mp
matplotlib.use("Agg")


class TimingCallback(BaseCallback):
    """
    Logs per-iteration:
      - time/rollout_time: seconds between _on_rollout_start and _on_rollout_end
      - time/train_time:   seconds between _on_rollout_end and the next _on_rollout_start
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._rollout_start = None
        self._last_rollout_end = None

    def _on_training_start(self) -> None:
        # Initialize a reference so the first measured train segment (after first rollout)
        # doesn't get a huge value from program start.
        self._last_rollout_end = time.time()

    def _on_rollout_start(self) -> None:
        now = time.time()
        # Training time is the gap since the last rollout ended
        if self._last_rollout_end is not None:
            train_time = now - self._last_rollout_end
            self.logger.record("time/train_time", float(train_time))
        self._rollout_start = now

    def _on_rollout_end(self) -> None:
        now = time.time()
        if self._rollout_start is not None:
            rollout_time = now - self._rollout_start
            self.logger.record("time/rollout_time", float(rollout_time))
        self._last_rollout_end = now

    def _on_step(self) -> bool:
        # Required by BaseCallback. Return True to continue training.
        return True


def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return {
            "commit": commit,
            "branch": branch
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown"
        }

# Version information
VERSION = {
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info()
}
def evaluate_with_rendering(config):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()
def train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler):
    seed = 1234
    np.random.seed(seed)

    # In run_train.py, before env is created:
    print(f"DEBUG: Config Object Foot Penalty: {config.env_params.reward_keys_and_weights.foot_force_penalty}") 
    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=log_dir,
        name_prefix=f"{config.algo or 'algo'}_ckpt"
        )


    timing_cb = TimingCallback()

    '''callbacks = CallbackList([timing_cb, checkpoint_cb])

    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=callbacks, progress_bar=True)
    final_path = os.path.join(log_dir, f"{config.algo or 'algo'}_final")
    model.save(final_path)
    if hasattr(model, "save_replay_buffer"):
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer.pkl"))'''
    
    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)
    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=[timing_cb, custom_callback], progress_bar=True)
    #model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=None, progress_bar=True)


    env.close()
    print("learning done!")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)


    DictionableDataclass.set_from_args(config, args, prefix="config.")


    log_dir = os.path.join("rl_train","results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    if args.flag_realtime_evaluate:
        evaluate_with_rendering(config)
    else:
        train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler)
    