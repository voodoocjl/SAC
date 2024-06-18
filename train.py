import matplotlib.pyplot as plt
from functions import *
from SAC import *
import time
import argparse
import random
import pickle
import copy

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

    for i in range(numbers):
        step_cnt = 0
        test_reward = 0        
                    
        agents = agents_test[i]
        _, sorted_indices=position_initial(agents)
               
        done_last = [agents[i].done for i in range(n_agents)]
        agent_observations = get_observations(agents,sorted_indices)
        
        # perform steps
        while not all(done_last):
            actions_np = SAC.select_action(agent_observations)  # 返回每个agent选择的动作      
            for a in agents:
                if done_last[agents.index(a)]==0:
                    a.vel_new = np.array([action_unnormalized(actions_np[agents.index(a)][0], maxSpeed, -maxSpeed),
                                                    action_unnormalized(actions_np[agents.index(a)][1], maxSpeed, -maxSpeed)])
                else:
                    a.vel_new=np.array([0,0])
                actions_np[agents.index(a)]=a.vel_new
                a.pos_new = a.pos_old + a.vel_new
            
            # get rewards of followers
            agent_rewards  = get_agent_rewards(agents,sorted_indices)
            
            # 任务结束
            c, short, num_cut, apl,d = get_values(agents)
            if (c == 1 and short == 0 and num_cut == 0) or step_cnt >= max_steps - 1:                
                for a in agents:
                    a.done = 1
            
            #get observations_next of followers
            agent_observations_next = get_observations(agents, sorted_indices)#可能会换目标智能体
            
            #update the historical values
            for a in agents:
                a.pos_old = a.pos_new
                a.vel_old = a.vel_new
            
            #push memory
            for j in range(len(agents)):
                if done_last[j]==0:                    
                    test_reward += agent_rewards[j]

            agent_observations=agent_observations_next
            done_last = [agents[i].done for i in range(n_agents)]
            step_cnt = step_cnt+1
        
        test_reward_list.append(test_reward)
    result = [round(np.mean(test_reward_list),2), round(np.std(test_reward_list),2)]
    return result

#param about SAC
batch_size = 256#每次更新网络参数，从回放池中抽取的样本数
replay_buffer_size = 5000#回放池的容量
n_actions=2#x，y轴的动作
n_states=(n_agents-1)*2#58,每个agent观察到的状态的维度，所有智能体的相对位置
before_training = 4
hidden_dim=128
SAC = SAC(n_states, n_actions,hidden_dim)
replay_buffer = ReplayBuffer(replay_buffer_size)
reward_record =[]
# reward_statics = []
Episode=[]
collide_record=[]
degree=[]
steps=[]
connectivity_rate=[]
num_shorts=[]
num_cuts=[]
# plot initial locations
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
'''Episode Run'''

parser = argparse.ArgumentParser()
parser.add_argument(    
    "-e", "--episode", type=int, default=10, help="episode number of model"
)
args = parser.parse_args()
max_episode = min(max_episode,args.episode)

ttot = 0
all_step = 0
all_term = 0
current = time.time()
t0 = current

with open('all_agents', 'rb') as outfile:
    agents_list = pickle.load(outfile)

period = 100
episode_idx = 1000
max_episode = 3000
SAC.load_models(episode_idx)
if episode_idx > 0:
    with open('memory_{}'.format(episode_idx), 'rb') as outfile:
        replay_buffer = pickle.load(outfile)

# test_agents(20)

while episode_idx<max_episode:
    setup_seed(episode_idx)

    tbegin = time.time()
    step_cnt = 0
    reward = 0
    step_connect=max_steps
    '''initialize the environment'''
    # reset the agent group   
        
    agents = agents_list[episode_idx]
    _, sorted_indices=position_initial(agents)   
    
    '''plot the initial positions'''
    # for a in agents:
    #     ax.scatter(a.pos_new[0],a.pos_new[1], c='r', marker='o')
    
    done_last = [agents[i].done for i in range(n_agents)]
    agent_observations = get_observations(agents,sorted_indices)
    
    # perform steps
    while not all(done_last):#如果没有一个智能体完成任务. done 数组中至少有一个元素为 False 时为 True，否则为 False。
        # 1. get agent actions and new positions        
        actions_np = SAC.select_action(agent_observations)  # 返回每个agent选择的动作      
        for a in agents:
            if done_last[agents.index(a)]==0:
                a.vel_new = np.array([action_unnormalized(actions_np[agents.index(a)][0], maxSpeed, -maxSpeed),
                                                  action_unnormalized(actions_np[agents.index(a)][1], maxSpeed, -maxSpeed)])
            else:
                a.vel_new=np.array([0,0])
            actions_np[agents.index(a)]=a.vel_new
            a.pos_new = a.pos_old + a.vel_new
        
        # get rewards of followers
        agent_rewards  = get_agent_rewards(agents,sorted_indices)
        
        # 任务结束
        c, short, num_cut, apl,d = get_values(agents)
        if (c == 1 and short == 0 and num_cut == 0) or step_cnt >= max_steps - 1:
            all_step += step_cnt
            all_term += 1
            for a in agents:
                a.done = 1
        
        #get observations_next of followers
        agent_observations_next = get_observations(agents, sorted_indices)#可能会换目标智能体
        
        #update the historical values
        for a in agents:
            a.pos_old = a.pos_new
            a.vel_old = a.vel_new
        
        #push memory
        for j in range(len(agents)):
            if done_last[j]==0:
                  replay_buffer.push(agent_observations[j], agents[j].vel_new, agent_rewards[j], agent_observations_next[j],agents[j].done)#均是numpy类型
                  reward+=agent_rewards[j]
        
        #update_network
        if len(replay_buffer) > before_training * batch_size :
            SAC.update_parameters(replay_buffer, batch_size)
        agent_observations=agent_observations_next
        
        # plot new positions
        # ax.clear()    # clear old points
        # for a in agents:
        #     ax.scatter(a.pos_new[0],a.pos_new[1], c='r', marker='o')

        # ax.text(-map_size[0],map_size[1],str(int(n_agents))+' agents', color='r', fontsize=12)
        # ax.text(-map_size[0]+150,map_size[1],str(int(step_cnt))+' steps', color='k', fontsize=12)
        # #set axis
        # ax.set_xlim(-map_size[0], map_size[0])
        # ax.set_ylim(-map_size[1], map_size[1])
        # plot
        # plt.pause(0.001)

        done_last = [agents[i].done for i in range(n_agents)]
        step_cnt = step_cnt+1

    
    if episode_idx != 0 and episode_idx % period == 0:
        current = time.time()
        t_100 = current - t0
        t0 = current
        rewards = reward_record[episode_idx-period :]
        reward_statics = [round(np.mean(rewards),2), round(np.std(rewards),2)]
        test_rewards = test_agents(20)

        with open('reward_stat.txt', 'a') as file:
            # Write data to the file
            file.write(str(reward_statics) + '\n')

        with open('reward_test.txt', 'a') as file:
            # Write data to the file
            file.write(str(test_rewards) + '\n')

        print("*" * 10)
        print(f"Elapsed time: {t_100:.2f}")
        print(f"avr step: {all_step / all_term:.2f}")
        print('Episode,agent_reward', episode_idx, reward_statics)
        print('Test_rewards', test_rewards)
        print("*" * 10)

    tend = time.time()
    ttot += tend - tbegin
    num_shorts.append(short)
    reward_record.append(reward)
    # reward_statics.append([torch.mean(reward), torch.var(reward)])
    Episode.append(episode_idx)
    degree.append(d)
    steps.append(step_cnt)
    connectivity_rate.append(c)
    num_cuts.append(num_cut)
    print('Episode,step_cnt,agent_reward,degree,connectivity_rate,num_cut,short', episode_idx,step_cnt, round(reward,2), d,
          c, num_cut, short)
    # print('Episode,agent_reward,num_cut', episode_idx,round([torch.mean(reward), torch.var(reward)],2), num_cut)
    episode_idx += 1
    if episode_idx == max_episode:
        SAC.save_models(max_episode)
        with open('memory_{}'.format(max_episode), 'wb') as outfile:
            pickle.dump(replay_buffer, outfile)
        print('Len of buffer: ', len(replay_buffer))        

print("###" * 10)

print(f"ttot: {ttot:.2f}")
print(f"avr step: {all_step / all_term:.2f}")

print("###" * 10)

file=open('reward30.txt','w')
for i in reward_record:
    file.write(str(round(i,2))+'\n')
file.close()

#画图，每一步对应的损失
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(Episode,reward_record,linewidth=1)
ax.set_xlabel('episode',fontsize=15)
ax.set_ylabel('reward',fontsize=15)
ax.tick_params(axis='x',labelsize=15)
ax.tick_params(axis='y',labelsize=15)
# ax.legend()
plt.savefig('reward{}.jpg'.format(max_episode),dpi=600)
# plt.show()



