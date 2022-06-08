import numpy as np
import sys
from duobuchang_ant import Ant
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimSun']


# 导入地图
MAP0 = np.zeros([20, 20])
MAP1 = MAP0.copy()
MAP1[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 16]] = 1
MAP1[[1, 1, 1, 1, 1, 1], [0, 4, 9, 11, 16, 17]] = 1
MAP1[[2, 2, 2, 2, 2, 2, 2], [2, 6, 7, 8, 12, 13, 16]] = 1
MAP1[[3, 3, 3, 3, 3, 3], [1, 3, 7, 9, 12, 17]] = 1
MAP1[[4, 4, 4, 4, 4, 4, 4], [4, 5, 7, 8, 11, 15, 19]] = 1
MAP1[[5, 5, 5, 5, 5, 5, 5, 5, 5], [1, 2, 9, 10, 11, 12, 15, 16, 17]] = 1
MAP1[[6, 6, 6, 6, 6, 6, 6, 6], [0, 4, 5, 8, 9, 11, 13, 14]] = 1
MAP1[[7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 1, 5, 6, 9, 13, 14, 16, 17]] = 1
MAP1[[8, 8, 8, 8, 8, 8, 8, 8], [3, 4, 7, 12, 14, 15, 16, 17]] = 1
MAP1[[9, 9, 9, 9, 9, 9, 9, 9], [0, 2, 3, 9, 12, 14, 15, 18]] = 1
MAP1[[10, 10, 10, 10, 10, 10, 10, 10], [0, 5, 8, 9, 13, 14, 15, 18]] = 1
MAP1[[11, 11, 11, 11, 11, 11, 11, 11, 11], [0, 4, 6, 7, 8, 11, 13, 15, 16]] = 1
MAP1[[12, 12, 12, 12, 12, 12, 12], [11, 12, 13, 15, 16, 17, 18]] = 1
MAP1[[13, 13, 13, 13, 13, 13, 13, 13], [7, 11, 12, 13, 15, 16, 17, 18]] = 1
MAP1[[14, 14, 14, 14, 14, 14, 14, 14], [2, 3, 12, 13, 15, 16, 17, 18]] = 1
MAP1[[15, 15, 15, 15, 15], [2, 3, 6, 7, 8]] = 1
MAP1[[16, 16, 16, 16], [11, 14, 17, 18]] = 1
MAP1[[17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17], [0, 1, 2, 3, 5, 7, 9, 10, 1, 14, 17, 18]] = 1
MAP1[[18, 18, 18, 18, 18, 18, 18, 18], [0, 2, 4, 5, 6, 12, 14, 15]] = 1
MAP1[[19, 19, 19], [0, 2, 3]] = 1

# 变量初始化
time_start = time.time()
m = 50
Alpha = 1
Beta = 7
G = 100
NC = 0
Rho = 0.3
p = 0.97  # 用于局部信息素更新的挥发系数
q0 = 0.5
mapsize = list(np.shape(MAP0))
Local_pheromone = np.zeros(mapsize)
Global_pheromone = np.ones(mapsize)
tau_max = 10
tau_min = 0.1
start_point = [0, 0]
end_point = [19, 19]
partial = 0.3
k1 = 0.6
k2 = 0.4
Q = 10
Q2 = 1  # 用于局部信息素更新
L_shortest = sys.float_info.max
SC_least = sys.float_info.max
L = []
Best_way = []
psi_least = sys.float_info.max
sisuo_num = []


# 初始信息素矩阵
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        if MAP1[i][j] == 1:
            Global_pheromone[i][j] = 0
        else:
            Global_pheromone[i][j] = tau_max
Pheromone = Global_pheromone.copy()

# 求出每个点到终点的距离
Distance = np.zeros(mapsize)
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        Distance[i][j] = ((i - end_point[0]) ** 2 + (j - end_point[1]) ** 2) ** 0.5

while NC <= G:
    for i in range(m):
        locals()['ant' + str(i)] = Ant()
        locals()['ant' + str(i)].start(MAP1, start_point) # 创建m只蚂蚁并放置在初始点
    a = np.arange(0, m)  #######
    moving_list = list(a)
    new_moving_list = moving_list.copy() # 用于记录删掉已完成循环的、死锁的蚂蚁后剩下的蚂蚁
    sisuo = []
    L_se = np.full(m, sys.float_info.max)
    SC_se = np.full(m, sys.float_info.max)

    # 所有蚂蚁完成一轮移动
    while True:
        for j in moving_list:
            if j in new_moving_list:
                locals()['ant' + str(j)].look(mapsize)
                if not locals()['ant' + str(j)].condition: # 判断是否死锁
                    new_moving_list.remove(j)
                    sisuo.append(j)
                else:
                    locals()['ant' + str(j)].move(Distance, Pheromone, q0, Alpha, Beta, end_point)
                    Local_pheromone = locals()['ant' + str(j)].update_local_pheromone(Local_pheromone,Q2,p,partial)
                    Pheromone = Global_pheromone + Local_pheromone
                    # 将信息素控制在一定范围内
                    for k in range(mapsize[0]):
                        for l in range(mapsize[1]):
                            if MAP1[k][l] == 0:
                                if Pheromone[k][l] > tau_max:
                                    Pheromone[k][l] = tau_max
                                elif Pheromone[k][l] < tau_min:
                                    Pheromone[k][l] = tau_min
                    if locals()['ant' + str(j)].tabu[end_point[0]][end_point[1]]: #判断蚂蚁是否到达终点
                        new_moving_list.remove(j)
                        L_se[j] = locals()['ant' + str(j)].compute_distance()
        if new_moving_list:
            continue
        else:
            break

    sisuo_num.append(len(sisuo))

    # 最佳通道的选择
    if L_shortest > min(L_se):
        L_shortest = min(L_se)
    if SC_least > min(SC_se):
        SC_least = min(SC_se)
    u = L_se / L_shortest
    v = SC_se / SC_least
    psi = k1 * u + k2 * v
    psi_min = min(psi)
    psi_list = list(psi)
    num_of_best = psi_list.index(psi_min)

    # 记录本次迭代最佳路径、本次最佳路径长度，更新最小psi
    Best_way.append(locals()['ant' + str(num_of_best)].path)
    L.append(L_se[num_of_best])

    # 全局信息素更新
    Delta_pheromone = np.zeros(mapsize)
    for i in locals()['ant' + str(num_of_best)].path: # 此时i应该是一个二维list
        for k in range(m):
            if i in locals()['ant' + str(k)].path:
                Delta_pheromone[i[0]][i[1]] = Q / L_se[k] + Delta_pheromone[i[0]][i[1]]
    Global_pheromone = (1 - Rho) * Global_pheromone + Rho * Delta_pheromone

    '''Delta_pheromone = np.zeros(mapsize)
    for k in range(m):
        for ppp in locals()['ant' + str(k)].path:
            Delta_pheromone[ppp[0]][ppp[1]] = Q / L_se[k] + Delta_pheromone[ppp[0]][ppp[1]]
    Global_pheromone = (1 - Rho) * Global_pheromone + Delta_pheromone'''

    # 增加迭代次数
    NC = NC + 1

# 寻找全局最佳路径
Num_of_best = L.index(min(L))
print(Best_way[Num_of_best])
print(L)
print(min(L))
print(sisuo_num)
print(sum(sisuo_num))
time_end = time.time()
t = time_end - time_start
print(t)

x = np.arange(NC)

plt.figure(figsize=(15,5))
ax1 = plt.subplot(122)
plt.xlabel('迭代次数',fontdict={'weight':'normal','size': 15})
plt.ylabel('最优路径长度',fontdict={'weight':'normal','size': 15})
plt.title('收敛曲线',fontdict={'weight':'normal','size': 17})
plt.plot(x, L)

# 画图
ax = plt.subplot(121)
plt.xlim(0,mapsize[1])
plt.ylim(0,mapsize[0])
obstacle_x = []
obstacle_y = []
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        if MAP1[i][j] == 1:
            obstacle_x.append([j,j+1])
            obstacle_y.append([mapsize[0]-i-1,mapsize[0]-i])
for k in range(len(obstacle_x)):
    ax.fill_between(obstacle_x[k],obstacle_y[k][0],obstacle_y[k][1],facecolor='black')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')
x = []
y = []
for l in range(len(Best_way[Num_of_best])):
    x.append(Best_way[Num_of_best][l][1]+0.5)
    y.append(mapsize[0]-Best_way[Num_of_best][l][0]-0.5)
plt.plot(x,y)
plt.title('最优路径',fontdict={'weight':'normal','size': 17})
plt.show()
