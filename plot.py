# this file is used to evaluate the performance of the ev2gym environment with various stable baselines algorithms.
#TODO 写我自己的算法主程序
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
# from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
import numpy as np  
from ev2gym.models.condogym import EV2Gym
from ev2gym.rl_agent.reward import Condo_reward, ProfitMax_TrPenalty_UserIncentives, SquaredTrackingErrorReward, profit_maximization
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ev2gym.rl_agent.state import V2G_Condo, PublicPST, V2G_profit_max, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import datetime


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="ddpg")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--proj', type=str, default="Condo")
    parser.add_argument('--name', type=str, default="test")
    parser.add_argument('--config_file', type=str, default="Condo")

    algorithm = parser.parse_args().alg
    device = parser.parse_args().device
    run_name = parser.parse_args().name
    proj = parser.parse_args().proj
    config_file = f"ev2gym/example_config_files/Condo.yaml"
    # load config
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    # set reward functions and state functions

    if config_file == "ev2gym/example_config_files/V2GProfitMax.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitMax'
    elif config_file == "ev2gym/example_config_files/PublicPST.yaml":
        reward_function = SquaredTrackingErrorReward
        state_function = PublicPST
        group_name = f'{config["number_of_charging_stations"]}cs_PublicPST'
    elif config_file == "ev2gym/example_config_files/V2GProfitPlusLoads.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives
        state_function = V2G_profit_max_loads
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitPlusLoads'
    elif config_file == "ev2gym/example_config_files/Condo.yaml":
        reward_function = Condo_reward
        state_function = V2G_Condo
        group_name = f'{config["number_of_charging_stations"]}cs_Condo'
    
    run_name += f'{algorithm}_{reward_function.__name__}_{state_function.__name__}'

    # run = wandb.init(project=proj,
    #                  sync_tensorboard=True,
    #                  group=group_name,
    #                  name=run_name,
    #                  save_code=True,
    #                  )

    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.condogym:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              'price_data': None
                              })

    env = gym.make('evs-v0')

    eval_log_dir = "./eval_logs/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(f"./saved_models/{group_name}", exist_ok=True)

    eval_callback = EvalCallback(env, best_model_save_path=eval_log_dir,
                                 log_path=eval_log_dir,
                                 eval_freq=config['simulation_length']*50,
                                 n_eval_episodes=10, deterministic=True,
                                 render=False)
###################################### set agent ####################################
# TODO 写我自己的算法
    if algorithm == "ddpg":
        model = DDPG("MlpPolicy", env, verbose=1,
                    learning_rate = 1e-3,
                    buffer_size = 1_000_000,  # 1e6
                    learning_starts = 100,
                    batch_size = 100,
                    tau = 0.005,
                    gamma = 0.99,                     
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "td3":
        model = TD3("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "sac":
        model = SAC("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "a2c":
        model = A2C("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "ppo":
        model = PPO("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    # elif algorithm == "tqc":
    #     model = TQC("MlpPolicy", env, verbose=1,
    #                 device=device, tensorboard_log="./logs/")
    # elif algorithm == "trpo":
    #     model = TRPO("MlpPolicy", env, verbose=1,
    #                  device=device, tensorboard_log="./logs/")
    # elif algorithm == "ars":
    #     model = ARS("MlpPolicy", env, verbose=1,
    #                 device=device, tensorboard_log="./logs/")
    # elif algorithm == "rppo":
    #     model = RecurrentPPO("MlpLstmPolicy", env, verbose=1,
    #                          device=device, tensorboard_log="./logs/")
    else:
        raise ValueError("Unknown algorithm")

    #################################### evaluation ######################################
    env = model.get_env()
    obs = env.reset()

    HORIZON = 20          # 未来时间步个数
    N_EV_SLOTS = 25       # EV 充电口数量
    EV_FEATURES = 4       # 每个口的特征数: [SOC, arrival/sim_len, departure/sim_len, req_soc]

    
    def flatten_obs(obs):
        """
        返回: flat (1D np.ndarray), names (list[str])
        兼容: dict / tuple / list / np.ndarray（含 VecEnv 批维度）
        """
        if isinstance(obs, dict):
            pieces, names = [], []
            for k, v in obs.items():
                v_arr, v_names = flatten_obs(v)
                pieces.append(v_arr)
                names += [f"{k}.{name}" for name in v_names]
            return np.concatenate(pieces) if pieces else np.array([]), names

        if isinstance(obs, (list, tuple)):
            pieces, names = [], []
            for i, v in enumerate(obs):
                v_arr, v_names = flatten_obs(v)
                pieces.append(v_arr)
                names += [f"{i}.{name}" for name in v_names]
            return np.concatenate(pieces) if pieces else np.array([]), names

        arr = np.asarray(obs)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        # 若是 VecEnv 且 batch=1，取第 0 个环境
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.ravel()
        names = [f"obs[{i}]" for i in range(arr.shape[0])]
        return arr.astype(float), names

    def make_fixed_labels():
        labels = []
        # 1) current step
        labels.append("t = current_step")
        # 2) power usage (t-1)
        labels.append("P_usage[t-1] (kW)")
        # 3) 后 20 个单位的电价
        for k in range(HORIZON):
            labels.append(f"price[t+{k}] ($/kWh)")
        # 4) 后 20 个单位的 demand（楼宇/系统需求）
        for k in range(HORIZON):
            labels.append(f"demand[t+{k}] (kW)")
        # 5) 后 20 个单位的 power limit（容量/上限）
        for k in range(HORIZON):
            labels.append(f"power_limit[t+{k}] (kW)")
        # 6) 25 个 EV 口的状态（每口 4 维）
        for i in range(N_EV_SLOTS):
            labels.append(f"ev[{i}].SOC (0~1)")
            labels.append(f"ev[{i}].arrival_time/sim_len")
            labels.append(f"ev[{i}].departure_time/sim_len")
            labels.append(f"ev[{i}].req_soc = required_energy/cap")
        return labels

    def _safe_any(x):
        try:
            import numpy as _np
            return bool(_np.any(x))
        except Exception:
            return bool(x)

    # ====== 采集、标注与绘图（替换原代码段）======
    obs_records = []
    col_names = None
    stats = []

    for t in range(config['simulation_length']):
        flat, names = flatten_obs(obs)

        # 第一次循环：按你指定的固定标签顺序套用；如维度不匹配则回退
        if col_names is None:
            fixed_labels = make_fixed_labels()
            if len(fixed_labels) == flat.size:
                col_names = fixed_labels
                print(f"[labels] 使用固定标签，维度 = {len(col_names)}")
            else:
                print(f"[warn] 固定标签维度 {len(fixed_labels)} 与观测维度 {flat.size} 不一致，改用 flatten_obs 的列名。")
                col_names = names

        obs_records.append(flat)

        # 你原来的动作（示例全零；若用策略改回 model.predict）
        # action, _states = model.predict(obs, deterministic=True)

        action = - np.ones(env.action_space.shape, dtype=float)[None, :]
        
        obs, reward, done, info = env.step(action)

        if _safe_any(done):
            stats.append(info)
            # VecEnv 会自动 reset

    # ========= 存储为 DataFrame / CSV =========
    obs_arr = np.stack(obs_records, axis=0)    # [T, D]
    import pandas as pd
    df = pd.DataFrame(obs_arr, columns=col_names)
    df.index.name = "t"
    df.to_csv("obs_history.csv", index=True)
    print("Saved to obs_history.csv, shape:", df.shape)
    print(df.head())

    # ========= 画一张总览图（所有维度做最小-最大归一化后叠在一张图上）=========
    import matplotlib.pyplot as plt
    import os

    def minmax_norm(a):
        mn = np.min(a, axis=0)
        mx = np.max(a, axis=0)
        span = np.where(mx > mn, mx - mn, 1.0)
        return (a - mn) / span

    norm_arr = minmax_norm(obs_arr)
    plt.figure()
    for i in range(norm_arr.shape[1]):
        plt.plot(norm_arr[:, i], label=col_names[i])
    plt.xlabel("t")
    plt.ylabel("normalized value")
    plt.title("All observation features (min-max normalized)")
    plt.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    # ========= 为每个维度各画一张单图，并保存 =========
    out_dir = "obs_plots"
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(col_names):
        plt.figure()
        plt.plot(df.index.values, df.iloc[:, i].values)
        plt.xlabel("t")
        plt.ylabel(name)
        plt.title(name)
        plt.tight_layout()
        fname = f"{i:03d}_" + "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name) + ".png"
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    print(f"Saved {len(col_names)} per-feature plots to ./{out_dir}/")