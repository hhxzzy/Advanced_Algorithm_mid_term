from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim # 行动的数量
        self.__device = device # 使用的设备
        self.__gamma = gamma # 对未来的折扣？

        self.__eps_start = eps_start # 初始epsilon值
        self.__eps_final = eps_final # 最终的epsilon值
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device) # 实际的Q网络
        self.__target = DQN(action_dim, device).to(device) # 展示固定的Q网络，作为目标
        if restore is None:
            self.__policy.apply(DQN.init_weights) # 对权重进行初始化
        else:
            self.__policy.load_state_dict(torch.load(restore)) # 加载已有的权重
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training: # 若在训练中，则epsilon需要递减，直到最小
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps: # 随着eps越来越小，随机选择的记录逐渐降低
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1) # 随机选择一个行动

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)

        values = self.__policy(state_batch.float()).gather(1, action_batch)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None: # 同步两个Q网络
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None: # 保存模型
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
