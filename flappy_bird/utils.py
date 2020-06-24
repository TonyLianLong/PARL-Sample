#coding:UTF-8
from contextlib import contextmanager
import cv2
import time

i = [0]

#捕获当前屏幕并resize成(84*84*1)的灰度图
def resizeBirdrToAtari(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    if i[0] % 10000 < 10:
        print(observation.sum())
    i[0] += 1
    return observation

@contextmanager
def trainTimer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST_Time:{}'.format(name, end - start))

