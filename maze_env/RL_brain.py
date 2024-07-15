import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        #创建二维表格,行是各种状态,列是各种行为
        self.actions=actions#表示一个列表,其中表示各种可用的行为
        #这里只是定义了若干列,,行数随着观测增加
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.eps=e_greedy
        self.gama=reward_decay
        self.lr=learning_rate
    def choose_action(self,observation):
        #boservation 参数会接受某一个时刻的State
        #检查state状态是否在qtable存在,如果不存在就添加进去   这个observation就是状态
        self.check_state_exists(observation)
        #根据q_table开始行为选择
        #让agent只能选择最优的行为,所谓嘴鸥就是action在state下对应的Q值最大
        #只选择最大值的效果一般,,要人为地增加一些随机性   if np.random.uniform() 在0~1之间选数

        if np.random.uniform()<self.eps:#0.9的概率选择最优,,0.1的概率随机游走
            #把state下的所有action中取出来
            state_actions=self.q_table.loc[observation,:]
            #选择最优,,当有多个相同最大值时,随机选择一个即可
            #注意这里关注的不是Q值,而是Q址对应的索引
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            action=np.random.choice(self.actions)
        return  action

    def learn(self,s,a,r,s_):
        #这里使用Q-learning
        #更新Q表中的每一个值   bellman方程   	Q(s,a)←Q(s,a)+lr[r+γmax(a')Q(s_,a_)−Q(s,a)]
        # s a r s_ 当前state,当前action,奖励,未来state
        #当前时刻的状态s已经在当前时刻选择行为的时候,也就是调用choose_action时,检查过了
        self.check_state_exists(s_)
        #预测就是查询Q_table中的数值
        q_predict=self.q_table.loc[s,a]
        #需要判断大钱回合是否进行到了最后
        if s_!='terminal':
            #计算r+γmax(a')Q(s_,a_)
            q_target=r+self.gama*self.q_table.loc[s_,:].max()
        else:
            q_target=r+self.gama*0
        #有了上述这两部分就可以更新Q(s,a)了
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict)
        pass

    def check_state_exists(self,state):
        #如果当前状态在qtable中不存在,就向qtable中添加一行
        if state not in self.q_table.index:
            #将这个状态添加到qtable中
            self.q_table=self.q_table._append(
                #dataframe是二维结构 其中每一个元素是一个一维的结构 Series
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )