from tqdm import tqdm
import random
import pickle
from functions import *

max_episode = 5000
valid_cnt = 0
agents_list = []
with tqdm(total=max_episode) as bar:
    while valid_cnt < max_episode:
        x = random.randint(10, 200) * ((-1) ** random.randint(1, 2))  # 15, 110
        y = random.randint(10, 200) * ((-1) ** random.randint(1, 2))  #

        agents = [Agent(x, y) for i in range(numAgents)]

        # 判断初始位置和最终位置是否离得太近，如果太近就重新生成初始位置和目标位置
        flag = position_initial(agents)
        if flag == 0:
            agents_list.append(agents)
            valid_cnt += 1            
            bar.update(1)
        else:
            continue

with open('all_agents', 'wb') as outfile:
    pickle.dump(agents_list, outfile)