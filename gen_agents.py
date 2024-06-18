from functions import *
import pickle

max_episode = 20000
episode_idx = 0
n_agents = 90
agents_list = []
while episode_idx < max_episode:       
    agents = [Agents() for i in range(n_agents)]
    flag,sorted_indices=position_initial(agents)
    if flag==1:
        continue
    else:
        agents_list.append(agents)
        episode_idx+=1
    if episode_idx % 1000 == 0:
        print("Number: ", episode_idx)

with open('all_agents', 'wb') as outfile:
    pickle.dump(agents_list, outfile)

