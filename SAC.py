import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from params import *
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import TensorDataset

# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

# ****************************************************


class ReplayBuffer:
    """经验池,记录智能体当前状态矩阵,动作,奖励,下一个状态矩阵, 完成目标情况"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # state: [122]
        # action: [2]
        # next_state: [122,]

        # print(
        #     f"buffer push: state,{state.shape}, action,{action.shape}, next_state,{next_state.shape}"
        # )
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # state & next_state:[batch,122], action:[batch,2]
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def weights_init_(m):
    """初始化神经网络的权重"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    """target net的权重更新基于Q-net的权重,进行动量更新"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """直接将source的权重复制到target"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def action_unnormalized(action, high, low):
    """动作数值范围转换函数(-1,1) -> [LOW, HIGH]"""
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action


class QNetwork(nn.Module):
    """奖励评估网络,负责根据状态和动作,预测奖励值"""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        #  self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        self.forward_t = 0
        self.forward_cnt = 0

    def forward(self, state, action):
        # print(f"critic forward: {state.shape}, {action.shape}")
        t1 = time.time()
        x_state_action = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        # x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear3_q1(x1)

        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        # x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear3_q2(x2)
        self.forward_t += time.time() - t1
        self.forward_cnt += 1

        return x1, x2


class PolicyNetwork(nn.Module):
    """策略动作网络,根据状态,预估最佳动作"""

    def __init__(
        self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2
    ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        self.forward_t = 0
        self.forward_cnt = 0

    def forward(self, state):
        """动作[Vx,Vy]的均值和标准差的对数"""
        # print(f"policy forward: {state.shape}")
        t1 = time.time()
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        self.forward_t += time.time() - t1
        self.forward_cnt += 1

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        """根据动作[Vx,Vy]的均值和标准差的进行正态分布采样"""
        # 正态分布采样
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # x_t = mean + eps * std, eps是不依赖与mean和std的正态分布采样
        x_t = normal.rsample()
        # tanh限制action的范围为[-1,1]
        action = torch.tanh(x_t)
        # tanh-log_prob, 重新计算概率分布
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


class SAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        gamma=0.99,
        tau=1e-2,
        alpha=0.2,
        lr=0.0003,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr

        self.target_update_interval = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(
            torch.Tensor([action_dim]).to(self.device)
        ).item()
        print("entropy", self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        self.update_q_t = 0
        self.update_actor_t = 0
        self.update_alpha_t = 0
        self.update_tq_t = 0

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # state: [1,n, 2+4*n]
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            # _, _, action, _ = self.policy.sample(state)
            # 使用forward替代sample, 减少冗余计算, 避免无效的log_prob计算
            # Q: 为什么训练时的action直接使用均值,而不是分布采样
            action_mean, _ = self.policy.forward(state)
            action = torch.tanh(action_mean)
        action = action.detach().cpu().numpy()[0]
        # action: [30,2]
        return action

    def update_parameters(
        self,
        memory,
        batch_size,
        n_batches=4
    ):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = memory.sample(batch_size = n_batches * batch_size)       

        
        state = torch.FloatTensor(state_batch).to(self.device)  # [batch, 122]
        next_state = torch.FloatTensor(next_state_batch).to(
            self.device
        )  # [batch, 122]
        action = torch.FloatTensor(action_batch).to(self.device)  # [batch, 2]
        reward = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        
        data = TensorDataset(state, next_state, action, reward, done)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, pin_memory=False)
        t1 = time.time()
        # 更新Q-Net权重参数
        for x in data_loader:
            state_batch = x[0]  # [batch, 122]
            next_state_batch = x[1]
            action_batch = x[2]
            reward_batch = x[3]
            done_batch = x[4]

            with torch.no_grad():
                next_state_action, next_state_log_pi, _, _ = self.policy.sample(
                    next_state_batch
                )
                qf1_next_target, qf2_next_target = self.critic_target(
                    next_state_batch, next_state_action
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = reward_batch + (1 - done_batch) * self.gamma * (
                    min_qf_next_target
                )

            qf1, qf2 = self.critic(
                state_batch, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  #
            qf2_loss = F.mse_loss(qf2, next_q_value)  #
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
        t2 = time.time()
        self.update_q_t += t2 - t1

        # 更新Actor(policy)权重参数
        for x in data_loader:
            state_batch = x[0]
            pi, log_pi, mean, log_std = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            t3 = time.time()
            self.update_actor_t += t3 - t2

            # 更新alpha参数
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

            t4 = time.time()
            self.update_alpha_t += t4 - t3

        # 更新Target Net权重参数
        soft_update(self.critic_target, self.critic, self.tau)
        t5 = time.time()
        self.update_tq_t += t5 - t4

   
    # Save model parameters
    def save_models(self, episode_count):
        # torch.save(self.policy.state_dict(), 'model/' + str(episode_count)+'_policy_30.pth')
        # torch.save(self.critic.state_dict(), 'model/'  +str(episode_count)+ '_value_30.pth')
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy_optim.state_dict(),
            "log_alpha": self.log_alpha.data,
            "log_alpha_grad": self.log_alpha.grad,
            "alpha_optimizer": self.alpha_optim.state_dict()
            },             
            'model/' + str(episode_count)+'_policy_model_and_optimizer.pth'
        )
        torch.save({
            "model_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optim.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict()
            }, 
            'model/' + str(episode_count)+'_critic_model_and_optimizer.pth'
        )
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Load model parameters
    def load_models(self,episode_count):
        if episode_count > 0:
            checkpoint_p = torch.load('model/' + str(episode_count) + '_policy_model_and_optimizer.pth')
            checkpoint_c = torch.load('model/' + str(episode_count) + '_critic_model_and_optimizer.pth')
            self.policy.load_state_dict(checkpoint_p['model_state_dict'])
            self.critic.load_state_dict(checkpoint_c['model_state_dict'])
            self.critic_target.load_state_dict(checkpoint_c['critic_target_state_dict'])
            self.policy_optim.load_state_dict(checkpoint_p['optimizer_state_dict'])
            self.critic_optim.load_state_dict(checkpoint_c['optimizer_state_dict'])
            self.log_alpha.data = checkpoint_p['log_alpha']
            self.log_alpha.grad = checkpoint_p['log_alpha_grad']
            self.alpha_optim.load_state_dict(checkpoint_p['alpha_optimizer'])
            self.log_alpha = self.log_alpha.to(self.device)
            self.alpha = self.log_alpha.exp()
            
            print('***Models load***')

