# import networkx as nx
# import pandas as pd
import time
from math import sqrt

from params import *

# get magnitude of a vector


def get_magnitude(vector):
    """
    计算两点之间的距离
    勾股定理: c = sqrt(a^2+b^2)"""
    return np.linalg.norm(vector)


def position_initial(agents):
    """判断智能体之间的起始位置和目标位置是否太近,低于最短距离"""
    flag = 0

    for agent_idx in range(len(agents)):
        for j in range(len(agents)):
            if j != agent_idx:
                distance1 = get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                distance2 = get_magnitude(agents[agent_idx].target - agents[j].target)
                if distance1 <= min_dis or distance2 <= min_dis:
                    flag = 1
                    return flag
    return flag


"""define observation function"""


def get_agent_observations(agents):
    """获取智能体的状态矩阵

    Args:
        agents (Agent): 智能体

    Returns:
        np.array: agent_observations: shape = [n,2+n*4]
        每个智能体的观察信息:
        当前位置与目标位置的x轴距离, 当前位置与目标位置的y轴距离
        与其他可能发生碰撞(安全距离不足)智能体(包括本身)的x轴距离, y轴距离
        与其他可能发生碰撞(安全距离不足)智能体(包括本身)的x轴速度, y轴速度
    """
    agent_observations = np.zeros([numAgents, numAgents * 4 + 2])
    for agent_idx in range(len(agents)):
        if agents[agent_idx].done == 0:
            agent_observations[agent_idx][0] = (
                agents[agent_idx].target[0] - agents[agent_idx].pos_new[0]
            )
            agent_observations[agent_idx][1] = (
                agents[agent_idx].target[1] - agents[agent_idx].pos_new[1]
            )
            i = 0
            for j in range(len(agents)):
                distance = get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                if distance <= maxvel * 2:  # 存在碰撞风险
                    agent_observations[agent_idx][4 * i + 2] = (
                        agents[j].pos_new[0] - agents[agent_idx].pos_new[0]
                    )
                    agent_observations[agent_idx][4 * i + 3] = (
                        agents[j].pos_new[1] - agents[agent_idx].pos_new[1]
                    )
                    agent_observations[agent_idx][4 * i + 4] = agents[j].vel_new[0]
                    agent_observations[agent_idx][4 * i + 5] = agents[j].vel_new[1]
                    i += 1
    return agent_observations


class Reward:
    def __init__(self) -> None:
        self.reach_t = 0
        self.exceed_t = 0
        self.coll_t = 0

    def get_agent_rewards(self, agents):
        """根据智能体的状态计算奖励"""

        agent_rewards = np.zeros(numAgents)
        num_collide = 0
        num_arrive = 0
        for agent_idx in range(len(agents)):
            if agents[agent_idx].done == 0:
                t1 = time.time()
                a = agents[agent_idx]
                dis_tar_now = get_magnitude(a.target - a.pos_new)
                agent_rewards[agent_idx] += -dis_tar_now * 0.1  # 以此为惩罚，让智能体不断接近目标

                # 碰到边界的惩罚
                if (
                    a.pos_new[0] >= map_size[0]
                    or a.pos_new[0] <= -map_size[0]
                    or a.pos_new[1] >= map_size[1]
                    or a.pos_new[1] <= -map_size[1]
                ):
                    agent_rewards[agent_idx] -= 20
                    a.done = 1
                    a.collide = 1
                    num_collide += 1
                t2 = time.time()
                self.exceed_t += t2 - t1
                # 智能体之间发生碰撞
                for j in range(len(agents)):
                    if j != agents.index(a):
                        distance = get_magnitude(a.pos_new - agents[j].pos_new)
                        # 与其他智能体碰撞：这一步的距离小于安全距离，或者两步之中路线有交叉且交点距离两者的起点相等
                        if distance <= min_safe_dis:
                            agent_rewards[agent_idx] -= 20
                            num_collide += 1
                            a.done = 1
                            a.collide = 1
                        else:
                            len1 = get_magnitude(a.pos_new - a.pos_old)
                            len2 = get_magnitude(agents[j].pos_new - agents[j].pos_old)
                            m1_x = (a.pos_new[0] + a.pos_old[0]) / 2
                            m1_y = (a.pos_new[1] + a.pos_old[1]) / 2
                            m2_x = (agents[j].pos_new[0] + agents[j].pos_old[0]) / 2
                            m2_y = (agents[j].pos_new[1] + agents[j].pos_old[1]) / 2
                            if (
                                len1 - len2 <= min_safe_dis
                                and sqrt((m1_x - m2_x) ** 2 + (m1_y - m2_y) ** 2)
                                <= min_safe_dis
                            ):
                                agent_rewards[agent_idx] -= 20
                                num_collide += 1
                                a.done = 1
                                a.collide = 1
                t3 = time.time()
                self.coll_t += t3 - t2
                # 到达目标
                if dis_tar_now <= dis_target and a.collide == 0:
                    agent_rewards[agent_idx] += 100
                    a.done = 1
                    num_arrive += 1
                t4 = time.time()
                self.reach_t += t4 - t3
        # print(f"reach_t:{reach_t:.2f}, exceed_t:{exceed_t:.2f}, coll_t:{coll_t:.2f}")
        return agent_rewards, num_collide, num_arrive


# 计算奖励
def get_agent_rewards(agents):
    """根据智能体的状态计算奖励"""
    reach_t = 0
    exceed_t = 0
    coll_t = 0

    agent_rewards = np.zeros(numAgents)
    num_collide = 0
    num_arrive = 0
    for agent_idx in range(len(agents)):
        if agents[agent_idx].done == 0:
            t1 = time.time()
            a = agents[agent_idx]
            dis_tar_now = get_magnitude(a.target - a.pos_new)
            agent_rewards[agent_idx] += -dis_tar_now * 0.1  # 以此为惩罚，让智能体不断接近目标

            # 碰到边界的惩罚
            if (
                a.pos_new[0] >= map_size[0]
                or a.pos_new[0] <= -map_size[0]
                or a.pos_new[1] >= map_size[1]
                or a.pos_new[1] <= -map_size[1]
            ):
                agent_rewards[agent_idx] -= 20
                a.done = 1
                a.collide = 1
                num_collide += 1
            t2 = time.time()
            exceed_t += t2 - t1
            # 智能体之间发生碰撞
            for j in range(len(agents)):
                if j != agents.index(a):
                    distance = get_magnitude(a.pos_new - agents[j].pos_new)
                    # 与其他智能体碰撞：这一步的距离小于安全距离，或者两步之中路线有交叉且交点距离两者的起点相等
                    if distance <= min_safe_dis:
                        agent_rewards[agent_idx] -= 20
                        num_collide += 1
                        a.done = 1
                        a.collide = 1
                    else:
                        len1 = get_magnitude(a.pos_new - a.pos_old)
                        len2 = get_magnitude(agents[j].pos_new - agents[j].pos_old)
                        m1_x = (a.pos_new[0] + a.pos_old[0]) / 2
                        m1_y = (a.pos_new[1] + a.pos_old[1]) / 2
                        m2_x = (agents[j].pos_new[0] + agents[j].pos_old[0]) / 2
                        m2_y = (agents[j].pos_new[1] + agents[j].pos_old[1]) / 2
                        if (
                            len1 - len2 <= min_safe_dis
                            and sqrt((m1_x - m2_x) ** 2 + (m1_y - m2_y) ** 2)
                            <= min_safe_dis
                        ):
                            agent_rewards[agent_idx] -= 20
                            num_collide += 1
                            a.done = 1
                            a.collide = 1
            t3 = time.time()
            coll_t += t3 - t2
            # 到达目标
            if dis_tar_now <= dis_target and a.collide == 0:
                agent_rewards[agent_idx] += 100
                a.done = 1
                num_arrive += 1
            t4 = time.time()
            reach_t += t4 - t3
    # print(f"reach_t:{reach_t:.2f}, exceed_t:{exceed_t:.2f}, coll_t:{coll_t:.2f}")
    return agent_rewards, num_collide, num_arrive
