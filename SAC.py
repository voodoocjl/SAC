import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from params import *
from torch.distributions import Normal
from torch.optim import Adam
import math

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
    """
    向量乘标量
    """
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
        self.linear4_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        #  self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)

        self.r_W1_q1 = torch.zeros((hidden_dim, state_dim + action_dim))
        self.r_W2_q1 = torch.zeros((hidden_dim, hidden_dim))
        self.r_W3_q1 = torch.zeros((1, hidden_dim))

        self.r_W1_q2 = torch.zeros((hidden_dim, state_dim + action_dim))
        self.r_W2_q2 = torch.zeros((hidden_dim, hidden_dim))
        self.r_W3_q2 = torch.zeros((1, hidden_dim))

        self.m_W1_q1 = torch.zeros((hidden_dim, state_dim + action_dim))
        self.m_W2_q1 = torch.zeros((hidden_dim, hidden_dim))
        self.m_W3_q1 = torch.zeros((1, hidden_dim))

        self.m_W1_q2 = torch.zeros((hidden_dim, state_dim + action_dim))
        self.m_W2_q2 = torch.zeros((hidden_dim, hidden_dim))
        self.m_W3_q2 = torch.zeros((1, hidden_dim))

        self.apply(weights_init_)
        self.forward_t = 0
        self.forward_cnt = 0

    def forward(self, state, action):
        
        # print(f"critic forward: {state.shape}, {action.shape}")
        t1 = time.time()
        self.x_state_action = torch.cat([state, action], 1)

        self.x1_q1 = F.relu(self.linear1_q1(self.x_state_action))
        self.x2_q1 = F.relu(self.linear2_q1(self.x1_q1))
        # x1 = F.relu(self.linear3_q1(x1))
        q1 = self.linear4_q1(self.x2_q1)

        self.x1_q2 = F.relu(self.linear1_q2(self.x_state_action))
        self.x2_q2 = F.relu(self.linear2_q2(self.x1_q2))
        # x2 = F.relu(self.linear3_q2(x2))
        q2 = self.linear4_q2(self.x2_q2)
        self.forward_t += time.time() - t1
        self.forward_cnt += 1

        return q1, q2 

    def get_critic_delta(self, x1, x2, q):
        return 2 * (x1 - q), 2 * (x2 - q)
    

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

        self.epsilon = 1e-6
        self.tr12 = 0
        self.te12 = 0
        self.tr23 = 0
        self.te23 = 0

        self.r_W1 = torch.zeros((hidden_dim, state_dim))
        self.r_W2 = torch.zeros((hidden_dim, hidden_dim))
        self.r_W_mean = torch.zeros((action_dim, hidden_dim))
        self.r_W_logstd = torch.zeros((action_dim, hidden_dim))

        self.m_W1 = torch.zeros((hidden_dim, state_dim))
        self.m_W2 = torch.zeros((hidden_dim, hidden_dim))
        self.m_W_mean = torch.zeros((action_dim, hidden_dim))
        self.m_W_logstd = torch.zeros((action_dim, hidden_dim))


    def forward(self, state):
        """动作[Vx,Vy]的均值和标准差的对数"""
        # print(f"policy forward: {state.shape}")
        t1 = time.time()
        self.state = state
        self.x1 = F.relu(self.linear1(state))
        self.x2 = F.relu(self.linear2(self.x1))
        mean = self.mean_linear(self.x2)
        log_std = self.log_std_linear(self.x2)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        self.forward_t += time.time() - t1
        self.forward_cnt += 1

        return mean, log_std

    def sample(self, state, train, epsilon=1e-6):
        """根据动作[Vx,Vy]的均值和标准差的进行正态分布采样"""
        # 正态分布采样
        t1 = time.time()
        mean, log_std = self.forward(state)
        t2 = time.time()
        if train:
            self.tr12 += t2-t1
        else:
            self.te12 += t2-t1


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


        t3 = time.time()
        if train:
            self.tr23 += t3-t2
        else:
            self.te23 += t3 - t2
        return action, log_prob, mean, log_std, x_t

    def get_policy_delta(self, action, mean, log_std, x_t, epsilon, alpha):
        """
            action: [256,2]
            mean: [256,2]
            epsilon: 标量
            alpha: 标量
        """
        alpha_d_256 = alpha

        action_s = action * action
        one_minus_action_s = 1 - action_s
        ookinafakutaa = (2 * action * one_minus_action_s)/(one_minus_action_s + 1e-6)
        sample_eps_times_std = (x_t - mean)

        delta_mean = alpha_d_256 * ookinafakutaa
        delta_std = alpha_d_256 * (ookinafakutaa * sample_eps_times_std - 1)

        return delta_mean, delta_std    
    

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
        self.t12 = 0
        self.t23 = 0
        self.t34 = 0
        self.t45 = 0
        self.t56 = 0
        self.t67 = 0
        self.t78 = 0
        self.t89 = 0
        self.t910 = 0
        self.t1011 = 0
        self.t1112 = 0
        self.t1213 = 0
        self.t1314 = 0

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # state: [1,n, 2+4*n]
        if eval == False:
            action, _, _, _, _ = self.policy.sample(state, False)
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
    ):
        t1 = time.time()
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = memory.sample(batch_size=1)

        state_batch = torch.FloatTensor(state_batch).to(self.device)  # [batch, 122]
        next_state_batch = torch.FloatTensor(next_state_batch).to(
            self.device
        )  # [batch, 122]
        action_batch = torch.FloatTensor(action_batch).to(self.device)  # [batch, 2]
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        t2 = time.time()
        self.t12 += t2 - t1
        t3 = 0
        t4 = 0
        # 更新Q-Net权重参数
        with torch.no_grad():

            next_state_action, next_state_log_pi, mean, _, _ = self.policy.sample(
                next_state_batch, True
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            t3 = time.time()
            self.t23 += t3 - t2
            ##### 加减乘
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (
                min_qf_next_target
            )
            t4 = time.time()
            self.t34 += t4 - t3
            #####

            
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        t5 = time.time();
        self.t45 += t5 - t4
        qf1_loss = F.mse_loss(qf1, next_q_value)  #
        qf2_loss = F.mse_loss(qf2, next_q_value)  #
        qf_loss = qf1_loss + qf2_loss
        t6 = time.time()
        self.t56 += t6 - t5

        ###################
        ### pytorch 自动 ###
        ###################
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        ###################
        ###     手动     ###
        ###################
        # self.critic.backward(qf1, qf2, next_q_value)

        t7 = time.time()
        self.t67 += t7 - t6

        # 更新Actor(policy)权重参数
        pi, log_pi, mean, log_std, x_t = self.policy.sample(state_batch, True)
        t8 = time.time()
        self.t78 += t8 - t7
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        t9 = time.time()
        self.t89 += t9 - t8
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        t10 = time.time()
        self.t910 += t10 - t9

        ###################
        ### pytorch 自动 ###
        ###################
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
     

        t11 = time.time()
        self.t1011 += t11 - t10

        # 更新alpha参数
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        t12 = time.time()
        self.t1112 += t12 - t11
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        t13 = time.time()
        self.t1213 += t13 - t12

        # 更新Target Net权重参数
        soft_update(self.critic_target, self.critic, self.tau)
        t14 = time.time()
        self.t1314 += t14 - t13

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
            self.alpha = self.log_alpha.exp()
            
            print('***Models load***')
