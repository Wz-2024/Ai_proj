import os

# 设置图片路径
image_path = r"E:/尚学堂ai/cats_and_dogs/kaggle_Dog&Cat"

# 检查文件是否存在
if os.path.exists(image_path):
    print("文件存在")
else:
    print("文件不存在")

