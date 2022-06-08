import numpy as np
import sys
from my_ant import Ant
import time
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MultipleLocator
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimSun']


def find_passby(start, end):
    passby = []
    if start[0] != end[0]:
        k = (start[1] - end[1]) / (start[0] - end[0])
        p1 = []
        p2 = []
        if start[0] < end[0]:
            p1 = start
            p2 = end
        elif start[0] > end[0]:
            p1 = end
            p2 = start  # p1为左边的点，p2为右边的点
        x_position = np.arange(0.5, p2[0] - p1[0] + 0.5, 1)
        if k > 0:
            y_position = k * x_position + 0.5
            a = math.ceil(y_position[0])
            for i in range(0, a):
                passby.append([p1[0], p1[1] + i])
            l = len(y_position)
            for j in range(1, l):
                b = math.floor(y_position[j - 1])
                c = math.ceil(y_position[j])
                for h in range(b, c):
                    passby.append([p1[0] + j, p1[1] + h])
            d = math.floor(y_position[-1])
            e = math.ceil(p2[1] - p1[1] + 0.5)
            for i in range(d, e):
                passby.append([p2[0], p1[1] + i])
        elif k < 0:
            y_position = k * x_position - 0.5
            a = math.floor(y_position[0])
            for i in range(0, a, -1):
                passby.append([p1[0], p1[1] + i])
            l = len(y_position)
            for j in range(1, l):
                b = math.ceil(y_position[j - 1])
                c = math.floor(y_position[j])
                for h in range(b, c, -1):
                    passby.append([p1[0] + j, p1[1] + h])
            d = math.ceil(y_position[-1])
            e = math.floor(p2[1] - p1[1] - 0.5)
            for i in range(d, e, -1):
                passby.append([p2[0], p1[1] + i])
        elif k == 0:
            u = min(start[0], end[0])
            v = max(start[0], end[0])
            for i in range(u, v + 1):
                passby.append([i, start[1]])
    else:
        a = min(start[1], end[1])
        b = max(start[1], end[1])
        for i in range(a, b + 1):
            passby.append([start[0], i])
    return passby


def find_middle(start, end):
    middle = []
    if end[0] != start[0]:
        for mx in range(start[0], end[0]): # 这里没有把最后一个点end放进去
            my = (mx - start[0]) * (end[1] - start[1]) / (end[0] - start[0]) + start[1]
            if abs(my - int(my)) < 0.000000000001:
                middle.append([mx, int(my)])
    else:
        for my in range(start[1], end[1]):
            middle.append([start[0], my])
    return middle


# 导入地图
MAP0 = np.zeros([30, 30])
MAP1 = MAP0.copy()
MAP1[[1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 6, 7, 10, 11, 26, 27]] = 1
MAP1[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 6, 7, 10, 11, 22, 23, 26, 27]] = 1
MAP1[[3, 3, 3, 3, 3, 3, 3, 3], [10, 12, 13, 14, 17, 18, 19, 20]] = 1
MAP1[[4, 4, 4, 4, 4, 4], [2, 4, 14, 21, 22, 23]] = 1
MAP1[[5, 5, 5, 5, 5], [1, 4, 17, 18, 22]] = 1
MAP1[[6, 6, 6, 6, 6, 6, 6], [1, 4, 6, 7, 8, 11, 28]] = 1
MAP1[[7, 7, 7, 7, 7, 7, 7, 7], [2, 6, 7, 8, 11, 22, 26, 27]] = 1
MAP1[[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [6, 7, 8, 11, 12, 13, 14, 15, 25, 26, 27]] = 1
MAP1[[9], [27]] = 1
MAP1[[10, 10, 10, 10, 10, 10, 10, 10, 10], [2, 6, 7, 16, 17, 18, 22, 27, 29]] = 1
MAP1[[11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11], [2, 3, 10, 11, 12, 16, 17, 18, 22, 23, 26]] = 1
MAP1[[12, 12, 12, 12, 12, 12, 12, 12, 12, 12], [2, 6, 7, 10, 11, 16, 17, 18, 21, 22]] = 1
MAP1[[13, 13, 13, 13, 13, 13], [3, 4, 7, 17, 18, 24]] = 1
MAP1[[14, 14, 14, 14], [5, 6, 7, 26]] = 1
MAP1[[15, 15, 15, 15, 15, 15, 15, 15], [6, 7, 12, 13, 14, 22, 23, 26]] = 1
MAP1[[16, 16, 16, 16, 16], [7, 10, 17, 18, 26]] = 1
MAP1[[17, 17, 17, 17, 17, 17, 17, 17], [2, 9, 10, 11, 14, 17, 18, 29]] = 1
MAP1[[18], [14]] = 1
MAP1[[19], [20]] = 1
MAP1[[20, 20, 20, 20, 20], [6, 7, 20, 26, 27]] = 1
MAP1[[21, 21, 21, 21, 21, 21, 21, 21, 21, 21], [2, 5, 6, 7, 10, 11, 22, 23, 26, 27]] = 1
MAP1[[22, 22, 22, 22], [10, 11, 13, 14]] = 1
MAP1[[23, 23, 23, 23, 23, 23], [16, 17, 18, 21, 22, 23]] = 1
MAP1[[24, 24, 24], [1, 4, 14]] = 1
MAP1[[25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25], [1, 4, 6, 8, 10, 11, 17, 18, 19, 21, 24]] = 1
MAP1[[26, 26, 26, 26, 26, 26], [2, 8, 20, 24, 26, 27]] = 1
MAP1[[27, 27, 27, 27], [2, 3, 11, 17]] = 1
MAP1[[28, 28, 28, 28], [5, 10, 11, 25]] = 1
MAP1[[29, 29, 29, 29], [5, 10, 11, 25]] = 1

# 变量初始化
time_start = time.process_time()
m = 50
Alpha = 1.5
Beta = 7
Beta1 = Beta
epsilon = 0.5  # ?
# epsilon1 = epsilon
eta = 0.7
G = 100
NC = 0
Rho = 0.7
Rho_min = 0.3  # ?
delta = 0.7
tau0 = 1
c = 2  # ?
gamma = 0.85
Q = 1
tau_min = 0.1  # ?
tau_max = 10  # ?
T = 5  # ?
mapsize = list(np.shape(MAP0))
Pheromone = np.ones(mapsize)
start_point = [0, 0]
end_point = [29,29]
L_shortest = sys.float_info.max
L = []
Best_way = []
sisuo_num = []

# 求出每个点到终点的距离
Distance = np.zeros(mapsize)
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        Distance[i][j] = ((i - end_point[0]) ** 2 + (j - end_point[1]) ** 2) ** 0.5

# 设置初始信息素
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        if MAP1[i][j] == 1:
            Pheromone[i][j] = 0
        else:
            small_0 = min(start_point[0], end_point[0])
            large_0 = max(start_point[0], end_point[0])
            small_1 = min(start_point[1], end_point[1])
            large_1 = max(start_point[1], end_point[1])
            if small_0 <= i <= large_0 and small_1 <= j <= large_1:
                Pheromone[i][j] = c
            else:
                Pheromone[i][j] = tau0
'''                
# 求初始信息素
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        if MAP1[i][j] == 1:
            Pheromone[i][j] = 0
        else:
            num_of_barrier = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if 0 <= i + k <= mapsize[0] - 1 and 0 <= j + l <= mapsize[1] - 1:
                        if MAP1[i+k][j+l] == 1:
                            num_of_barrier = num_of_barrier + 1
            Pheromone[i][j] = q / (num_of_barrier + 1)
'''

# 求每个栅格邻近的栅格总数
A = np.zeros(mapsize)
for i in range(mapsize[0]):
    for j in range(mapsize[1]):
        if i == 0 or i == mapsize[0] - 1 or j == 0 or j == mapsize[1] - 1:
            if [i, j] == [0, 0] or [i, j] == [0, mapsize[1] - 1] or [i, j] == [mapsize[0] - 1, 0] or \
                    [i, j] == [mapsize[0] - 1, mapsize[1] - 1]:
                A[i, j] = 3
            else:
                A[i, j] = 5
        else:
            A[i, j] = 8

while NC <= G:
    for i in range(m):
        locals()['ant' + str(i)] = Ant()
        locals()['ant' + str(i)].start(MAP1, start_point)
    a = np.arange(0, m)
    moving_list = list(a)
    new_moving_list = moving_list.copy()  # 用于记录删掉已完成循环的、死锁的蚂蚁后剩下的蚂蚁
    sisuo = []
    L_se = np.full(m, sys.float_info.max)
    q0 = delta * (G - NC) / G
    '''
    if NC < G * 0.5:
        epsilon = epsilon1 * (G - 2 * NC) / G
    else:
        epsilon = 0
'''

    while True:
        for j in moving_list:
            if j in new_moving_list:
                locals()['ant' + str(j)].look(mapsize)
                if not locals()['ant' + str(j)].condition:
                    locals()['ant' + str(j)].jump_sisuo_look(mapsize,MAP1)
                    if not locals()['ant' + str(j)].condition:
                        new_moving_list.remove(j)
                        sisuo.append(j)
                    else:
                        locals()['ant' + str(j)].move(Pheromone, Alpha, Beta, epsilon, end_point, Distance, A, mapsize,
                                                      q0)
                        if locals()['ant' + str(j)].tabu[end_point[0]][end_point[1]]:
                            new_moving_list.remove(j)
                            L_se[j] = locals()['ant' + str(j)].compute_distance()
                else:
                    locals()['ant' + str(j)].move(Pheromone, Alpha, Beta, epsilon, end_point, Distance, A, mapsize, q0)
                    if locals()['ant' + str(j)].tabu[end_point[0]][end_point[1]]:
                        new_moving_list.remove(j)
                        L_se[j] = locals()['ant' + str(j)].compute_distance()
        if new_moving_list:
            continue
        else:
            break

    # 记录本次迭代最佳路径、本次最佳路径长度
    L_min = min(L_se)
    L_se_list = list(L_se)
    num_of_best = L_se_list.index(L_min)
    Best_way.append(locals()['ant' + str(num_of_best)].path)
    L.append(L_min)
    sisuo_num.append(len(sisuo))
    L_shortest = min(L)
    Num_of_best = L.index(min(L))

    # 信息素更新
    l_list = []
    L_se_list = list(L_se)
    for l_se in L_se:
        if l_se < 100000:
            l_list.append(l_se)
    l_best = min(l_list)
    num_of_best = L_se_list.index(l_best)
    l_worst = max(l_list)
    num_of_worst = L_se_list.index(l_worst)
    Delta_pheromone = np.zeros(mapsize)
    for k in range(m):
        if k not in sisuo:
            if k == num_of_best:
                for p in locals()['ant' + str(k)].path:
                    Delta_pheromone[p[0]][p[1]] = (1 + eta) * Q / L_se[k] + Delta_pheromone[p[0]][p[1]]
            elif k == num_of_worst:
                for p in locals()['ant' + str(k)].path:
                    Delta_pheromone[p[0]][p[1]] = (1 - eta) * Q / L_se[k] + Delta_pheromone[p[0]][p[1]]
            else:
                for p in locals()['ant' + str(k)].path:
                    Delta_pheromone[p[0]][p[1]] = Q / L_se[k] + Delta_pheromone[p[0]][p[1]]
    Pheromone = (1 - Rho) * Pheromone + Rho * Delta_pheromone
    for k in range(mapsize[0]):
        for lx in range(mapsize[1]):
            if MAP1[k][lx] == 0:
                if Pheromone[k][lx] > tau_max:
                    Pheromone[k][lx] = tau_max
                elif Pheromone[k][lx] < tau_min:
                    Pheromone[k][lx] = tau_min

    # Rho的选取
    if NC >= T-1:
        justice1 = True
        for i in range(T-1):
            if L[-i - 1] != L[-i - 2]:
                justice1 = False
        if justice1:
            if Rho * gamma >= Rho_min:
                Rho = Rho * gamma
            else:
                Rho = Rho_min

    # 最优路径的信息素强化
    if NC >= T - 1:
        justice2 = True
        for i in range(T-1):
            if L[-i - 1] == L_shortest:
                justice2 = False
        if justice2:
            for p in Best_way[Num_of_best]:
                Pheromone[p[0]][p[1]] = 10 * Q / L_shortest + Pheromone[p[0]][p[1]]
                if 1 <= p[0] < mapsize[0]-1:
                    if MAP1[p[0]+1][p[1]] == 0:
                        Pheromone[p[0]+1][p[1]] = 10 * Q / L_shortest + Pheromone[p[0]+1][p[1]]
                    if MAP1[p[0]-1][p[1]] == 0:
                        Pheromone[p[0]-1][p[1]] = 10 * Q / L_shortest + Pheromone[p[0]-1][p[1]]

    # 增加迭代次数
    NC = NC + 1

Num_of_best = L.index(min(L))
print(Best_way[Num_of_best])
print(L)
print(min(L))
print(sisuo_num)
print(sum(sisuo_num))

# 二次简化路径
way = []
for b in Best_way[Num_of_best]:
    way.append(b)
length_of_path = len(way)
aa = list(range(length_of_path))
for i in range(length_of_path):
    if i in aa and i < length_of_path - 2:
        for j in range(i+2,length_of_path):
            passby = find_passby(way[i],way[j])
            justice = True  # 判断是否有障碍物
            for k in passby:
                if MAP1[k[0]][k[1]] == 1:
                    justice = False
            if justice:
                if j-1 in aa:
                    aa.remove(j-1)
new_way = []
for a in aa:
    new_way.append(way[a])
new_L = 0
length = len(new_way)
for i in range(length - 1):
    new_L = new_L + ((new_way[i][0] - new_way[i + 1][0]) ** 2 + (new_way[i][1] - new_way[i + 1][1]) ** 2) ** 0.5
print(new_way)
print(new_L)

# 再一次简化路径
way1 = []
for r in range(len(new_way)-1):
    ww = find_middle(new_way[r], new_way[r + 1])
    for w in ww:
        way1.append(w)
way1.append(new_way[-1])
way1.reverse()
length_of_path1 = len(way1)
bb = list(range(length_of_path1))
for i in range(length_of_path1):
    if i in bb and i < length_of_path1 - 2:
        for j in range(i+2,length_of_path1):
            passby = find_passby(way1[i],way1[j])
            justice = True  # 判断是否有障碍物
            for k in passby:
                if MAP1[k[0]][k[1]] == 1:
                    justice = False
            if justice:
                if j-1 in bb:
                    bb.remove(j-1)
new_way1 = []
for b in bb:
    new_way1.append(way1[b])
new_way1.reverse()
new_L1 = 0
length1 = len(new_way1)
for i in range(length1 - 1):
    new_L1 = new_L1 + ((new_way1[i][0] - new_way1[i + 1][0]) ** 2 + (new_way1[i][1] - new_way1[i + 1][1]) ** 2) ** 0.5
print(new_way1)
print(new_L1)


time_end = time.process_time()
t = time_end - time_start
print(t)
x = np.arange(NC)


# 画图
plt.figure(figsize=(15,5))
ax1 = plt.subplot(122)
plt.xlabel('迭代次数',fontdict={'weight':'normal','size': 15})
plt.ylabel('最优路径长度',fontdict={'weight':'normal','size': 15})
plt.title('收敛曲线',fontdict={'weight':'normal','size': 17})
plt.plot(x, L)


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
for l in range(len(new_way1)):
    x.append(new_way1[l][1]+0.5)
    y.append(mapsize[0]-new_way1[l][0]-0.5)
plt.plot(x,y)
plt.title('最优路径',fontdict={'weight':'normal','size': 17})
plt.show()
