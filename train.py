import argparse
import random
import time
from pathlib import Path
from functions import *
from SAC import *
import copy
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def test_agents(numbers):

    setup_seed(42)
    agents_test = copy.deepcopy(agents_list[-numbers:])
    test_reward_list = []
    test_arrive_list = []

    for i in range(numbers):
        step_cnt = 0
        test_reward = 0
        num_collide=0
        num_arrive=0
                    
        agents = agents_test[i]   
        done_last = [agents[i].done for i in range(numAgents)]
        agent_observations = get_agent_observations(agents)  # 获得每个智能体的观测信息
        
        # perform steps
        while not all(done_last):  # 如果还有智能体没有完成任务就继续执行本轮
        
            # get agent actions and new positions
            actions_np = SAC.select_action(agent_observations)  # 返回每个agent选择的动作
            for a in agents:
                if done_last[agents.index(a)] == 0:
                    a.vel_new = np.array(
                        [action_unnormalized(actions_np[agents.index(a)][0], maxSpeed, -maxSpeed), action_unnormalized(actions_np[agents.index(a)][1], maxSpeed, -maxSpeed)])
                else:
                    a.vel_new = np.array([0, 0])
                a.pos_new = a.pos_old + a.vel_new
                if a.collide == 1:
                    a.pos_new = np.array([-20000, -20000])  # 因碰撞而坠毁
            
            # get observations_next of followers
            agent_observations_next = get_agent_observations(agents)  # 返回的是一个follower_nums*4的数组
        

            # get rewards of followers
            # agent_rewards, collides, arrive = get_agent_rewards(agents)
            agent_rewards, collides, arrive = rew.get_agent_rewards(agents)
            num_collide += collides
            num_arrive += arrive

            # update the historical values
            for a in agents:
                a.pos_old = a.pos_new
                a.vel_old = a.vel_new

            if step_cnt >= max_steps - 1:
                for a in agents:
                    a.done = 1

            # push memory
            for j in range(len(agents)):
                if done_last[j] == 0:               
                    test_reward += agent_rewards[j]

            agent_observations=agent_observations_next
            done_last = [agents[i].done for i in range(numAgents)]
            step_cnt = step_cnt+1
        
        test_reward_list.append(test_reward)
        test_arrive_list.append(num_arrive)
    result = [round(np.mean(test_arrive_list),2), round(np.std(test_arrive_list),2), round(np.mean(test_reward_list),2), round(np.std(test_reward_list),2)]    
    return result

parser = argparse.ArgumentParser()

output_dir = 'model'

Path(output_dir).mkdir(exist_ok=True, parents=True)

SAC = SAC(n_states, n_actions, hidden_dim)
replay_buffer = ReplayBuffer(replay_buffer_size)
reward_record = []
Episode = []
arrive_record = []
collide_record = []

"""Episode Run"""

# max_episode = min(max_episode, episode)
episode_per_saved = 1000
suc_cnt = 0
fail_cnt = 0

period = 100

start = 0
episode_idx = start
max_episode = 2000
SAC.load_models(episode_idx)
if episode_idx > 0:
    with open('model/memory_{}'.format(episode_idx), 'rb') as outfile:
        replay_buffer = pickle.load(outfile)

# 提前生成好合法的agents

valid_cnt = 0
agents_list = []

with open('all_agents', 'rb') as outfile:
    agents_list = pickle.load(outfile)

rew = Reward()

# test_agents(50)

start_time = time.time()
t0 = start_time

while episode_idx<max_episode:
    setup_seed(episode_idx)

    #记录一轮中发生碰撞和到达目的地完成任务的情况
    num_collide=0
    num_arrive=0
    step_cnt = 0
    reward=0

    agents = agents_list[episode_idx % 5000]
    done_last = [agents[i].done for i in range(numAgents)]
    agent_observations = get_agent_observations(agents)  # 获得每个智能体的观测信息
    
    # perform steps
    while not all(done_last):  # 如果还有智能体没有完成任务就继续执行本轮
        debug_t1 = time.time()
        # get agent actions and new positions
        actions_np = SAC.select_action(agent_observations)  # 返回每个agent选择的动作
        for a in agents:
            if done_last[agents.index(a)] == 0:
                a.vel_new = np.array(
                    [action_unnormalized(actions_np[agents.index(a)][0], maxSpeed, -maxSpeed), action_unnormalized(actions_np[agents.index(a)][1], maxSpeed, -maxSpeed)])
            else:
                a.vel_new = np.array([0, 0])
            a.pos_new = a.pos_old + a.vel_new
            if a.collide == 1:
                a.pos_new = np.array([-20000, -20000])  # 因碰撞而坠毁
        
        # get observations_next of followers
        agent_observations_next = get_agent_observations(agents)  # 返回的是一个follower_nums*4的数组
       

        # get rewards of followers
        # agent_rewards, collides, arrive = get_agent_rewards(agents)
        agent_rewards, collides, arrive = rew.get_agent_rewards(agents)
        num_collide += collides
        num_arrive += arrive

        # update the historical values
        for a in agents:
            a.pos_old = a.pos_new
            a.vel_old = a.vel_new

        if step_cnt >= max_steps - 1:
            for a in agents:
                a.done = 1

        # push memory
        for j in range(len(agents)):
            if done_last[j] == 0:
                replay_buffer.push(
                    agent_observations[j],
                    agents[j].vel_new,
                    agent_rewards[j],
                    agent_observations_next[j],
                    agents[j].done,
                )  # 均是numpy类型
                reward += agent_rewards[j]

        # update_network
        # print(
        #     f"len(replay_buffer:{len(replay_buffer)}, before_training * batch_size:{before_training * batch_size}"
        # )
        if len(replay_buffer) > before_training * batch_size:
            SAC.update_parameters(replay_buffer, batch_size)
        agent_observations = agent_observations_next        
        
        done_last = [agents[i].done for i in range(numAgents)]
        step_cnt = step_cnt + 1

    
    reward_record.append(reward)
    Episode.append(episode_idx)
    arrive_record.append(num_arrive)
    collide_record.append(num_collide)
    print(
        f"{episode_idx}: agent_reward:{reward:.2f}, num_collide:{num_collide}, num_arrive:{num_arrive}",
    )

    episode_idx += 1
    if episode_idx % period == 0:
        current = time.time()
        t_100 = current - t0
        t0 = current

        arrive = arrive_record[episode_idx-start-period :]
        rewards = reward_record[episode_idx-start-period :]
        reward_statics = [round(np.mean(arrive),2), round(np.std(arrive),2), round(np.mean(rewards),2), round(np.std(rewards),2)]
        test_rewards = test_agents(50)

        with open('reward_stat.txt', 'a') as file:
            # Write data to the file
            file.write(str(reward_statics) + '\n')

        with open('reward_test.txt', 'a') as file:
            # Write data to the file
            file.write(str(test_rewards) + '\n')

        print("*" * 10)
        print(f"Elapsed time: {t_100:.2f}")        
        print('Episode,agent_reward', episode_idx, reward_statics)
        print('Test_rewards', test_rewards)
        print("*" * 10)

    if episode_idx % episode_per_saved == 0 or episode_idx >= max_episode:        
        SAC.save_models(episode_idx)
        with open('model/memory_{}'.format(episode_idx), 'wb') as outfile:
            pickle.dump(replay_buffer, outfile)
   

"""Show the run time"""
end_time = time.time()
print(
    f"总时间: {end_time - start_time:.2f} s",
)