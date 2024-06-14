import numpy as np
import torch
print(torch.__version__)
y1 = [10.5, 12.3, 14.7, 17.1, 19.5, 21.9, 24.4, 26.8, 29.2, 31.6, 34.0, 36.5, 38.9, 39.2, 39.5, 39.8, 40.0, 40.0, 40.0, 40.0]
y2 = [11.0, 13.4, 15.6, 17.8, 20.1, 22.3, 24.7, 27.0, 29.4, 31.7, 34.1, 36.3, 38.5, 39.0, 39.3, 39.6, 39.9, 40.0, 40.0, 40.0]
y3 = [10.2, 12.6, 15.0, 17.3, 19.7, 22.0, 24.4, 26.7, 29.1, 31.4, 33.7, 36.1, 38.4, 39.1, 39.4, 39.7, 39.9, 40.0, 40.0, 40.0]

z1 = [10.1, 12.5, 14.8, 17.2, 19.6, 21.9, 24.3, 26.6, 29.0, 31.3, 33.7, 36.0, 38.4, 39.0, 39.3, 39.6, 39.8, 40.0, 40.0, 40.0]
z2 = [10.3, 12.7, 15.0, 17.4, 19.8, 22.1, 24.5, 26.8, 29.2, 31.5, 33.9, 36.2, 38.6, 39.1, 39.4, 39.7, 39.9, 40.0, 40.0, 40.0]
z3 = [10.4, 12.8, 15.2, 17.5, 19.9, 22.2, 24.6, 26.9, 29.3, 31.6, 34.0, 36.3, 38.7, 39.2, 39.5, 39.8, 40.0, 40.0, 40.0, 40.0]

# 打印数据以检查
print("y1:", y1)
print("y2:", y2)
print("y3:", y3)
print("z1:", z1)
print("z2:", z2)
print("z3:", z3)

import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(10, 6))  # 创建第一个图形对象并设置大小
plt.plot(range(len(y1)), y1, color='green', label='5:5')
plt.plot(range(len(y2)), y2, color='red', label='6:4')
plt.plot(range(len(y3)), y3, color='yellow', label='7:3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Image')
plt.legend()
plt.savefig("loss_test.png")  # 保存损失曲线图
plt.show()  # 显示损失曲线图

# 绘制准确率曲线
plt.figure(figsize=(10, 6))  # 创建第二个图形对象并设置大小
plt.plot(range(len(z1)), z1, color='green', label='5:5')
plt.plot(range(len(z2)), z2, color='red', label='6:4')
plt.plot(range(len(z3)), z3, color='yellow', label='7:3')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Image')
plt.legend()
plt.savefig("accuracy_test.png")  # 保存准确率曲线图
plt.show()  # 显示准确率曲线图
