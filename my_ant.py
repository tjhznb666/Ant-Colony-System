import numpy as np
import random
import math


class Ant:
    def __init__(self):
        self.path = []
        self.tabu = []
        self.arrive = []
        self.pass_by = []
        self.position = []
        self.condition = True
        self.visual_field = []
        self.active_field = []

    def start(self, MAP1, start_point):
        self.tabu = MAP1.tolist()
        self.tabu[start_point[0]][start_point[1]] = 1
        self.position = start_point
        self.path.append(self.position)

    def look(self, mapsize):
        self.visual_field = []
        self.active_field = []
        # 求视野域
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= self.position[0] + i <= mapsize[0] - 1 and \
                        0 <= self.position[1] + j <= mapsize[1] - 1 :
                    self.visual_field.append([self.position[0] + i, self.position[1] + j])
        for o in self.visual_field:
            if self.tabu[o[0]][o[1]] == 0:
                self. active_field.append(o)
        # 判断是否死锁
        if not self.active_field:
            self.condition = False

    def jump_sisuo_look(self,mapsize,MAP1):
        self.condition = True
        self.active_field = []
        self.visual_field = []
        # 求视野域
        for i in range(-3, 4):
            for j in range(-3, 4):
                if 0 <= self.position[0] + i <= mapsize[0] - 1 and \
                        0 <= self.position[1] + j <= mapsize[1] - 1:
                    self.visual_field.append([self.position[0]+i,self.position[1]+j])
        self.visual_field.remove(self.position)
        # 求活动域
        for k in self.visual_field:
            if self.tabu[k[0]][k[1]] == 0:
                passby = self.find_passby(self.position, k)
                passby.remove(self.position)
                practice = True
                for l in passby:
                    if MAP1[l[0]][l[1]] == 1:
                        practice = False
                if practice:
                    self.active_field.append(k)
        # 判断是否死锁
        if not self.active_field:
            self.condition = False

    def move(self, Pheromone, Alpha, Beta, epsilon, end_point, Distance, A, mapsize, q0):
        eta = []
        # 求启发因子
        for m in self.active_field:
            if m == end_point:
                eta.append(1000)
            else:
                eta.append(1/Distance[m[0]][m[1]])
        # 计算安全度
        safety = []
        for m in self.active_field:
            l_sum = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if 0 <= m[0] + k <= mapsize[0] - 1 and 0 <= m[1] + l <= mapsize[1] - 1:
                        if self.tabu[m[0]+k][m[1]+l] == 1:
                            l_sum = l_sum + 1
            a = A[m[0]][m[1]]
            safety.append((a-l_sum)/a)
        # 选择移动目标
        ll = len(eta)
        D = []
        for e in range(ll):
            d = (eta[e] ** Beta) * (Pheromone[self.active_field[e][0]][self.active_field[e][1]] ** Alpha) * \
                (safety[e] ** epsilon)
            D.append(d)
        q = random.random()
        if q <= q0:
            dd = max(D)
            self.position = self.active_field[D.index(dd)]
        else:
            s = sum(D)
            DD = np.array(D) / s
            pcum = []
            p = 0
            for i in DD:
                p = p + i
                pcum.append(p)
            rand = random.random()
            select = [i for i in pcum if i < rand]
            self.position = self.active_field[len(select)]
        # 将选中的栅格加入禁忌表以及蚂蚁路线
        self.tabu[self.position[0]][self.position[1]] = 1
        self.path.append(self.position)

    def compute_distance(self):
        d = 0
        length = len(self.path)
        for i in range(length - 1):
            d = d + ((self.path[i][0] - self.path[i + 1][0]) ** 2 + (
                    self.path[i][1] - self.path[i + 1][1]) ** 2) ** 0.5
        return d

    def find_passby(self, start, end):
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
            a = min(start[1],end[1])
            b = max(start[1],end[1])
            for i in range(a, b+1):
                passby.append([start[0],i])
        return passby
