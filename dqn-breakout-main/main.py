# main函数是用来训练模型的，暂时看不懂具体在干什么

from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False # 是否进行渲染
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4 # 每4步更新一次policy的模型
TARGET_UPDATE = 10_000 # 每10000步同步一次target的模型
WARM_STEPS = 50_000
MAX_STEPS = 5_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX) # 保存训练好的模型

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device) # 创建一个环境，用于跑atari这个游戏
agent = Agent( # 创建一个agent
    env.get_action_dim(), # 游戏中动作的数量，一共有三个，分别是左右和不动
    device, # 训练使用的设备
    GAMMA,
    new_seed(),
    EPS_START, # epsilon的开始值
    EPS_END, # epsilon的最小值
    EPS_DECAY, # epsilon递减的
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device) # 用来记录agent的动作于结果之间的联系，用于后面神经网络的训练

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS # 前面是热身
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done) # 加入记忆池中

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE) # 使用之前的记忆进行学习

    if step % TARGET_UPDATE == 0: # 将policy的网络同步到Target中去
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join( # 保存当前模型
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
