import flappy_bird.game.wrapped_flappy_bird as game
import numpy as np
import pygame

#有个两种可选的行为,一种是不动,小鸟自己下落
#另一种是点击鼠标左键,向上飞一下
ACTIONS=2


def play_game():
    #启动游戏
    game_state=game.GameState()

    while True:
        a_t=np.zeros([ACTIONS])

        #array([1.,0.]表示下降,就是鼠标不点击
        #array([0.,1.])表示上升,表示点击一下

        #游戏刚开始默认下降
        a_t[0]=1

        #检测当前是否点击了鼠标
        for event in pygame.event.get():
            if event.type==pygame.MOUSEBUTTONDOWN:
                a_t=np.zeros([ACTIONS])
                a_t[1]=1
            else:
                pass

        #最重要的就是调用这个方法,,需要传入输入的Actions
        _,_,terminal=game_state.frame_step(a_t)




def main():
    play_game()

if __name__ == '__main__':
    main()