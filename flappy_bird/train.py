import numpy as np
from random import random, randint, sample
import flappy_bird.game.wrapped_flappy_bird as game
import argparse
import torch
import torch.nn as nn
from flappy_bird.src.deep_q_network import DeepQNetowrk
from flappy_bird.src.utils import resize_and_bgr2gray,image_to_tensor


# 读入超参数的函数方法
def get_args():
    parser = argparse.ArgumentParser('''用DQN训练你FlappyBird''')
    parser.add_argument('--image_size', type=int, default=84, help='表示图像的高和宽')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--gama', type=float, default=0.99)
    parser.add_argument('--initial_epsilon', type=float, default=0.1)
    parser.add_argument('--final_epsilon', type=float, default=1e-4)
    parser.add_argument('--num_iters', type=int, default=2000000 )
    parser.add_argument('--saved_path', type=str, default='trained_models')
    parser.add_argument('--replay_memory_size', type=int, default=50000, help='缓存的大小')
    args = parser.parse_args()
    return args






def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # 获取模型
    model = DeepQNetowrk()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # 设置损失函数
    loss_f = nn.MSELoss()
    # 游戏环境
    game_state = game.GameState()
    # 定义一开始的动作,其实就是do nothing
    action = torch.zeros(2, dtype=torch.float32)
    # [1,0]表示do nothing,[0,1]表示向上飞一下
    action[0] = 1
    # 让游戏开始运行
    image_data, reward, terminal = game_state.frame_step(action)
    # 对图像进行预处理
    # 改变大小,让彩色的变成黑白的
    image_data = resize_and_bgr2gray(image_data)
    # 处理好的数据当前是numpy类型的,,,需要转化为pytorch可用的Tensor张量类型
    image_data = image_to_tensor(image_data)

    if torch.cuda.is_available():
        image_data = image_data.cuda()
        model.cuda()

    # State状态其实是连续的4帧画面,,在一开始的时候,将状态的4帧全部设置成初始的第一帧画面
    # 其目的是让模型在一开始能正常运行
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # 进行多次迭代,反复训练
    replay_memory = []
    iter = 0
    while iter < opt.num_iters:

        # 让模型正向传播
        prediction = model(state)[0]

        # 初始化action
        # 初始化action
        action = torch.zeros(2, dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # 获得epsilon,让智能体探索超参数
        epsilon = opt.final_epsilon + (
                    (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        # 获得一个random()随机数
        u = random()
        # 这个random_action是一个标志位,,,标志是否采取随机动作
        random_action = u <= epsilon
        if random_action:
            print('采取随机动作')
            action_index = randint(0, 1)

        else:
            print('采取Q值最大的动作')
            action_index = torch.argmax(prediction).item()
        action[action_index] = 1

        # 与环境进行交互,,,得到下一个状态
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        if torch.cuda.is_available():
            image_data = image_data.cuda()

        # 这里的切片操作其实是将某一帧去掉
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)

        # 这是就有了transition,就是一次状态的转换,将他加入到replay_memory中去
        replay_memory.append([state, action, reward, next_state, terminal])
        # 如果replay_memory满了,就把最早添加进来的那一条删掉
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        # 从缓存中获取一个批次的数据
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.cat(tuple(action for action in action_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(next_state for next_state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # Q(S,A)
        current_prediction_batch = model(state_batch)
        # Q(S',A)
        next_prediction_batch = model(next_state_batch)

        # 准备y target 对于一个批次的
        # target=R+max(Q(S',a))
        y_batch = torch.cat(tuple(
            reward if terminal else reward + opt.gama * torch.max(prediction) for reward, terminal, prediction in
            zip(reward_batch, terminal_batch, next_prediction_batch
                )))
        # 模型的预测值 prediction
        # 这里的action_batch是one-hot编码,,,相乘相加就是action对应的某个模型预测的Q值

        q_value = torch.sum(current_prediction_batch * action_batch.view(-1, 2), dim=1)
        optimizer.zero_grad()
        loss = loss_f(q_value, y_batch)
        loss.backward()
        optimizer.step()

        #清缓存
        torch.cuda.empty_cache()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon: {}, Reward: {}, Q-value: {}".format(
            iter + 1, opt.num_iters, action, loss, epsilon, reward, torch.max(prediction)))

        # 间隔一段时间,保存模型
        if (iter + 1) % 1000000  == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
