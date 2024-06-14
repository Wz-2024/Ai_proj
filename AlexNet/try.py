import os

# 设置图片路径
image_path = r"E:\尚学堂ai\cats_and_dogs\kaggle_Dog&Cat\kaggle_Dog&Cat\validation\cats\6.jpg"

# 检查文件是否存在
if os.path.exists(image_path):
    print("文件存在")
else:
    print("文件不存在")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图片并显示
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')  # 隐藏坐标轴
plt.show()
