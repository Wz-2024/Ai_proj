from maze_env import Maze
from RL_brain import QLearningTable


# 有了maze环境,,有了智能体brain. 接下来完成RL的流程,不断交互

def update():
    # episode 表示回合
    for episode in range(100):
        # 每回合开始,让agent处于maze当中的一个位置,比如左上角的位置,(1,1)
        # 不同的位置就是不同的state状态
        observation = env.reset()
        while True:
            # 渲染env()环境
            env.render()

            # 让Agent基于观测状态进行行为选择
            action = agent.choose_action(str(observation))
            # 让环境根据给出的action行为返回'下一时刻的状态,reward奖励,done标志位'
            observation_, reward, done = env.step(action)
            # 更新Qtable
            agent.learn(str(observation), action, reward, str(observation_))

            observation = observation_
            if done:
                break
    print('end of game')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    agent = QLearningTable(actions=list(range(env.n_actions)))

    # 环境间隔多长时间,就调用一次update
    # update在每一次调用的过程中,agent和env进行交互,也就是agent进行行为选择
    env.after(100,update)
    env.mainloop()


