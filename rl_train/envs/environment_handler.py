import numpy as np
import json
from stable_baselines3.common.vec_env import SubprocVecEnv
from myosuite.utils import gym
from rl_train.utils.data_types import DictionableDataclass
import os
from rl_train.train.train_configs.config import TrainSessionConfigBase
import dataclasses

class EnvironmentHandler:
    ALGO_MAP = {
        "ppo":  ("PPO",  "ppo_params",  True),   # on-policy
        "a2c":  ("A2C",  "a2c_params",  True),
        "trpo": ("TRPO", "trpo_params", True),
        "sac":  ("SAC",  "sac_params",  False),  # off-policy
        "td3":  ("TD3",  "td3_params",  False),
        "ddpg": ("DDPG", "ddpg_params", False),
    }

    @staticmethod
    def _to_kwargs(params):
        if params is None:
            return {}
        if dataclasses.is_dataclass(params):
            try:
                from rl_train.utils.data_types import DictionableDataclass
                return DictionableDataclass.to_dict(params)
            except Exception:
                return dataclasses.asdict(params)
        if isinstance(params, dict):
            return params
        if hasattr(params, "dict") and callable(params.dict):
            return params.dict()
        if hasattr(params, "__dict__"):
            return {k: v for k, v in vars(params).items() if not k.startswith("_")}
        return {}

    @staticmethod
    def _detect_algo(config):
        # 1) Explicit flag wins if present and recognized
        algo = getattr(config, "algo", None)
        if isinstance(algo, str) and algo.lower() in EnvironmentHandler.ALGO_MAP:
            return algo.lower()

        # 2) Otherwise infer from a non-empty params block
        for key, (_, param_attr, _) in EnvironmentHandler.ALGO_MAP.items():
            params = getattr(config, param_attr, None)
            if params is None:
                continue
            try:
                from rl_train.utils.data_types import DictionableDataclass
                as_dict = DictionableDataclass.to_dict(params)
            except Exception:
                as_dict = dict(params.__dict__) if hasattr(params, "__dict__") else {}
            if isinstance(as_dict, dict) and len(as_dict) > 0:
                return key

        raise ValueError("Could not determine RL algorithm. Set config.algo or provide a non-empty '*_params' block.")


    @staticmethod
    def _algo_spec(config):
        algo_key = EnvironmentHandler._detect_algo(config)
        cls_name, params_attr, is_on_policy = EnvironmentHandler.ALGO_MAP[algo_key]
        return algo_key, cls_name, params_attr, is_on_policy


    @staticmethod
    def create_environment(config, is_rendering_on:bool, is_evaluate_mode:bool = False):

        ref_data_dict = EnvironmentHandler.load_reference_data(config)
    
        # Base gym.make arguments
        gym_make_args = {
            'seed': config.env_params.seed,
            'model_path': config.env_params.model_path,
            'env_params': config.env_params,
            'is_evaluate_mode': is_evaluate_mode
        }
        
        # Add reference_data only if it exists
        if ref_data_dict is not None:
            gym_make_args['reference_data'] = ref_data_dict
        
        try:
            if is_rendering_on or config.env_params.num_envs == 1:
                print(f"{config.env_params.env_id=}")
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                if is_rendering_on:
                    env.mujoco_render_frames = True
                config.env_params.num_envs = 1

                # Only adjust n_steps for on-policy algos that actually use it
                try:
                    _, _, params_attr, is_on_policy = EnvironmentHandler._algo_spec(config)
                    if is_on_policy:
                        algo_params = getattr(config, params_attr, None)
                        if isinstance(algo_params, dict):
                            if "n_steps" in algo_params and "batch_size" in algo_params:
                                algo_params["n_steps"] = algo_params["batch_size"]
                        elif algo_params is not None and hasattr(algo_params, "n_steps") and hasattr(algo_params, "batch_size"):
                            algo_params.n_steps = algo_params.batch_size
                except Exception:
                    pass


            else:
                env = SubprocVecEnv([lambda: (gym.make(config.env_params.env_id, 
                                                    **gym_make_args)).unwrapped 
                                for _ in range(config.env_params.num_envs)])
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e
        return env

    @staticmethod
    def load_reference_data(config):
        # Check if config has reference_data_path attribute
        print("===================================================================")
        if not hasattr(config.env_params, 'reference_data_path'):
            print("No reference data path provided.")
            print("===================================================================")
            return None
            
        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("===================================================================")
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("===================================================================")
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(config.env_params.reference_data_path, allow_pickle=True)
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    def get_config_type_from_session_id(session_id):
        # from rl_train.envs import myo_leg_18_reward_per_step
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        # Create appropriate config based on env_id
        print(f"session_id: {session_id}")
        if session_id == 'myoAssistLeg-v0':
            return TrainSessionConfigBase
        elif session_id in ['myoAssistLegImitation-v0']:
            return ImitationTrainSessionConfig
        elif session_id == 'myoAssistLegImitationExo-v0':
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")
        

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        config_file_path = config_path
        with open(config_file_path, 'r') as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        
        from rl_train.envs import myoassist_leg_imitation
        from rl_train.utils import learning_callback
        if isinstance(config, ImitationTrainSessionConfig):
            custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
                original_reward_weights=config.env_params.reward_keys_and_weights,
                auto_reward_adjust_params=config.auto_reward_adjust_params,
            )
        else:
            custom_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
            )

        return custom_callback
    @staticmethod
    def get_stable_baselines3_model(config: TrainSessionConfigBase, env, trained_model_path: str | None = None):
        import stable_baselines3
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy

        # Debug print
        algo_key, cls_name, params_attr, is_on_policy = EnvironmentHandler._algo_spec(config)
        print(f"[ALG DETECT] algo='{getattr(config, 'algo', None)}' -> {algo_key} | class={cls_name} | params={params_attr}")


        # Figure out algo + params first
        algo_key, cls_name, params_attr, is_on_policy = EnvironmentHandler._algo_spec(config)

        # Choose policy class
        if is_on_policy:
            # Keep your custom on-policy policies
            if config.env_params.env_id in ["myoAssistLegImitationExo-v0"]:
                policy_class = HumanExoActorCriticPolicy
                print("Using HumanExoActorCriticPolicy")
            else:
                policy_class = HumanActorCriticPolicy
                print("Using HumanActorCriticPolicy")
        else:
            # Off-policy (SAC/TD3/DDPG) -> use SB3 built-in MlpPolicy
            policy_class = "MlpPolicy"
            print(f"Using built-in {policy_class} for off-policy algo '{algo_key}'")


        # Choose algo + params
        algo_key, cls_name, params_attr, _ = EnvironmentHandler._algo_spec(config)
        if algo_key == "trpo":
            from sb3_contrib import TRPO as algo_cls
        else:
            algo_cls = getattr(stable_baselines3, cls_name)

        algo_params = getattr(config, params_attr, None)
        if algo_params is None:
            raise ValueError(f"Missing parameter block '{params_attr}' for algo '{algo_key}'")

        algo_kwargs = EnvironmentHandler._to_kwargs(algo_params)

        # After algo detection:
        if is_on_policy:
            policy_kwargs = EnvironmentHandler._to_kwargs(getattr(config, "policy_params", None))
        else:
            # Off-policy (SAC/TD3/DDPG): use optional offpolicy block, or let SB3 defaults apply
            policy_kwargs = EnvironmentHandler._to_kwargs(getattr(config, "offpolicy_policy_params", None)) or None



        if trained_model_path is not None:
            print(f"Loading trained model ({cls_name}) from {trained_model_path}")
            custom_objects = {"policy_class": policy_class} if is_on_policy else None
            model = algo_cls.load(trained_model_path, env=env, custom_objects=custom_objects)
        elif getattr(config.env_params, "prev_trained_policy_path", None):
            prev_path = config.env_params.prev_trained_policy_path
            print(f"Loading previous trained policy ({cls_name}) from {prev_path}")
            custom_objects = {"policy_class": policy_class} if is_on_policy else None
            model = algo_cls.load(prev_path, env=env, custom_objects=custom_objects)
            # Optionally reset parts of the network after load if your custom policy supports it
            if hasattr(model, "policy") and hasattr(config, "policy_params"):
                cps = config.policy_params.custom_policy_params
                if hasattr(model.policy, "reset_network"):
                    model.policy.reset_network(
                        reset_shared_net=getattr(cps, "reset_shared_net_after_load", False),
                        reset_policy_net=getattr(cps, "reset_policy_net_after_load", False),
                        reset_value_net=getattr(cps, "reset_value_net_after_load", False),
                    )
        else:
            print(f"Instantiating {cls_name} with '{params_attr}'")
            model = algo_cls(
                policy=policy_class,
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=2,
                **algo_kwargs,
            )
        return model

    @staticmethod
    def updateconfig_from_model_policy(config, model):
        pass
        # config.policy_info.extractor_policy_net = f"{model.policy.mlp_extractor.policy_net}"
        # config.policy_info.extractor_value_net = f"{model.policy.mlp_extractor.value_net}"
        # config.policy_info.action_net = f"{model.policy.action_net}"
        # config.policy_info.value_net = f"{model.policy.value_net}"
        # config.policy_info.ortho_init = f"{model.policy.ortho_init}"
        # config.policy_info.share_features_extractor = f"{model.policy.share_features_extractor}"
