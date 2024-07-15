import numpy as np
a=[1,2,2,0]
a=np.array(a)
# action=np.random.choice(a[a==np.max(a)].index)

tmp=[2,3,4]
# tmp=np.array(tmp)
idx=np.random.choice( tmp[tmp==max(tmp)].index)




re=np.random.choice(a)
print(re)
# print(action)