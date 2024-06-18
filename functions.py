import networkx as nx
import numpy as np
import pandas as pd
from params import *
from math import sqrt
# get magnitude of a vector
import networkx as nx
import random
import matplotlib.pyplot as plt
import math
def get_magnitude(vector):
    return np.linalg.norm(vector)


def  get_values(agents):
    G = nx.Graph()
    G.add_nodes_from((range(n_agents)))
    short=0
    for agent_idx in range(n_agents):
        for j in range(n_agents):
            if j!=agent_idx:
                distance = get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                if j == agents[agent_idx].node1 or j == agents[agent_idx].node2:
                    if R_min<= distance <= communication_R:
                        G.add_edge(agent_idx, j)
                if distance <min_d:
                    short+=1
    num_cut=0
    if nx.is_connected(G):
        num_cut=len(list(nx.articulation_points(G)))
    # 计算最大连通子图的节点数量n
    largest_component = max(nx.connected_components(G), key=len)
    # 计算连通率
    connectivity_rate = len(largest_component) / len(G)
    d = nx.degree(G)
    degrees = [x[1] for x in d]
    if connectivity_rate== 1 and short == 0 and num_cut == 0:
        APL=nx.average_shortest_path_length(G)
    else:
        APL=-1
    return connectivity_rate,short,num_cut,APL,sum(degrees)/n_agents

def targets(agents,sorted_indices):
    # 第一个目标点是编号比自己小的离自己最近的智能体agent1，第2个目标点是agent1的目标智能体或者以agent1为目标的编号比自己小的智能体
    for i in range(3, n_agents):  # 按照到中心点的距离从小到大排序的节点列表，从第4个点开始往前找
        if agents[sorted_indices[i]].done == 0:
            distance = []
            for j in range(i):
                distance.append(get_magnitude(agents[sorted_indices[i]].pos_new - agents[sorted_indices[j]].pos_new))
            # 使用enumerate获取元素及其索引
            indexed_list1 = list(enumerate(distance))
            # 按照元素从小到大的顺序对indexed_list进行排序
            sorted_indexed_list1 = sorted(indexed_list1, key=lambda x: x[1])
            # 构建索引列表
            sorted_indices1 = [index for index, _ in sorted_indexed_list1]
            agents[sorted_indices[i]].node1 = sorted_indices[sorted_indices1[0]]

            distance2 = []
            idx2 = []
            for j in range(i):
                if agents[sorted_indices[j]].node1 == agents[sorted_indices[i]].node1 or agents[sorted_indices[j]].node2 == \
                        agents[sorted_indices[i]].node1:
                    distance2.append(get_magnitude(agents[sorted_indices[i]].pos_new - agents[sorted_indices[j]].pos_new))
                    idx2.append(sorted_indices[j])
            d1 = get_magnitude(
                agents[sorted_indices[i]].pos_new - agents[agents[agents[sorted_indices[i]].node1].node1].pos_new)
            d2 = get_magnitude(
                agents[sorted_indices[i]].pos_new - agents[agents[agents[sorted_indices[i]].node1].node2].pos_new)
            distance2.append(d1)
            distance2.append(d2)
            idx2.append(agents[agents[sorted_indices[i]].node1].node1)
            idx2.append(agents[agents[sorted_indices[i]].node1].node2)
            sorted_indices2 = sorted(range(len(distance2)), key=lambda k: distance2[k])
            sorted_index2 = [idx2[k] for k in sorted_indices2]
            agents[sorted_indices[i]].node2 = sorted_index2[0]

def  position_initial(agents):
    G = nx.Graph()
    G.add_nodes_from((range(n_agents)))
    for agent_idx in range(n_agents):
        for j in range(n_agents):
            if j != agent_idx:
                distance = get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)
                if distance<min_d:
                    return 1,0
                if min_d <= distance<=R_max:
                    G.add_edge(agent_idx, j)
    if nx.is_connected(G)==0:
        return 1, 0
    # 最接近中心位置的节点是永远不动的
    center = np.zeros(2)
    for agent_idx in range(len(agents)):
        center += agents[agent_idx].pos_new
    center = center / n_agents
    dis_center = []
    for agent_idx in range(len(agents)):
        dis_center.append(get_magnitude(agents[agent_idx].pos_new - center))
    # 使用enumerate获取元素及其索引
    indexed_list = list(enumerate(dis_center))
    # 按照元素从小到大的顺序对indexed_list进行排序
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    # 构建索引列表
    sorted_indices = [index for index, _ in sorted_indexed_list]  # 节点的标号，到中心点的距离从小到大排序

    agents[sorted_indices[0]].done = 1
    # agents[min_index].static = 1  # 该点一直静止
    agents[sorted_indices[0]].node1 = sorted_indices[1]
    agents[sorted_indices[0]].node2 = sorted_indices[2]
    agents[sorted_indices[1]].node1 = sorted_indices[0]
    agents[sorted_indices[1]].node2 = sorted_indices[0]
    agents[sorted_indices[2]].node1 = sorted_indices[0]
    agents[sorted_indices[2]].node2 = sorted_indices[1]

    return 0 ,sorted_indices# 这次初始化是有效的



'''define observation function'''
def get_observations(agents,sorted_indices):
    agent_observations = np.zeros([n_agents, (n_agents-1)*2])
    targets(agents, sorted_indices)
    for agent_idx in range(len(agents)):
        if agents[agent_idx].done==0:
            a=agents[agent_idx]
            agent_observations[agent_idx][0] = agents[a.node1].pos_new[0]-agents[agent_idx].pos_new[0]
            agent_observations[agent_idx][1] = agents[a.node1].pos_new[1]-agents[agent_idx].pos_new[1]
            agent_observations[agent_idx][2] = agents[a.node2].pos_new[0] - agents[agent_idx].pos_new[0]
            agent_observations[agent_idx][3] = agents[a.node2].pos_new[1] - agents[agent_idx].pos_new[1]
            i=0
            p=sorted_indices.index(agent_idx)
            for j in sorted_indices[:p]:
                if get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new)<=communication_R and j!=a.node1 and j!=a.node2:
                        agent_observations[agent_idx][i+4] = agents[j].pos_new[0] - agents[agent_idx].pos_new[0]
                        agent_observations[agent_idx][i+5] = agents[j].pos_new[1] - agents[agent_idx].pos_new[1]
                        i+=2
    return agent_observations

#计算奖励
def get_agent_rewards(agents,sorted_indices):
    agent_rewards = np.zeros(n_agents)
    for agent_idx in range(len(agents)):
        a = agents[agent_idx]
        if a.done==0:
          #1.减少不必要的节点移动。
          agent_rewards[agent_idx] +=-1

          #2.对于目标智能体，连通的奖励
          d_node1 = get_magnitude(agents[agent_idx].pos_new - agents[a.node1].pos_new)
          d_node2 = get_magnitude(agents[agent_idx].pos_new - agents[a.node2].pos_new)
          d_last_node1 = get_magnitude(agents[agent_idx].pos_old - agents[a.node1].pos_old)
          d_last_node2 = get_magnitude(agents[agent_idx].pos_old - agents[a.node2].pos_old)
          p = sorted_indices.index(agent_idx)
          if d_node1>=R_min and d_node2>=R_min:
              agent_rewards[agent_idx] += (d_last_node1 - d_node1) + (d_last_node2 - d_node2)
              if d_node1<=communication_R and d_node2<=communication_R:
                  if agents[a.node1].done==1 and agents[a.node2].done==1:
                      n=0
                      for j in sorted_indices[:p]:
                          if get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new) < min_d:
                               n += 1
                      if n == 0:
                          a.done=1
                          agent_rewards[agent_idx] += 40
                  else:
                      agent_rewards[agent_idx] += 1

          else:
              agent_rewards[agent_idx] += -10
          #避撞,如果和通信范围内除了目标智能体过近，就给与惩罚，否则就奖励
          n = 0
          for j in sorted_indices[:p]:
              if get_magnitude(agents[agent_idx].pos_new - agents[j].pos_new) < min_d:
                  n += 1
          if n!=0:
                agent_rewards[agent_idx] += -10
          else:#没有距离过小的
                agent_rewards[agent_idx] += 1



    return agent_rewards
