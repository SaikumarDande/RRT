import cv2
import numpy as np
import random
import math

class Node:
    def __init__(self, x, y):
        self.parent = None
        self.x = x
        self.y = y
        self.childs = []
        self.distance = 0

class RRTMap:
    
    def __init__(self, start, goal, MapDimensions):
        self.start = start
        self.goal = goal
        self.MapDimensions = MapDimensions
        self.Mapw, self.Maph = self.MapDimensions[0], self.MapDimensions[1]
        self.MapImg = np.ones([self.Maph, self.Mapw, 3], np.uint8)*255
        self.MapWindowName = "RRT Path Planning"
        self.nodeRad=2
        self.nodeThickness=-1
        #Colours
        self.Black = (0, 0, 0)
        self.Blue = (255, 0, 0)
        self.Green = (0, 255, 0)
        self.Red = (0, 0, 255)
        self.White = (255, 255, 255)
        #Obstacles
        self.obstacles = []
        self.dummyMap = None
        
    def drawMap(self, obstacles):
        self.drawNode('G')
        self.drawNode('S')
        self.drawObs(obstacles)
        self.dummyMap =  self.MapImg.copy()
        cv2.imshow(self.MapWindowName, self.MapImg)
        cv2.waitKey(1)
    
    def getMap(self):
        return self.MapImg.copy()
    
    def refreshMap(self):
        cv2.imshow(self.MapWindowName, self.MapImg)
        cv2.waitKey(1)
    
    def drawNode(self, nodeType, coords=None):
        if nodeType == 'G':
            cv2.circle(self.MapImg, (self.goal[0],self.goal[1]), 5, self.Blue, self.nodeThickness)
        if nodeType == 'S':
            cv2.circle(self.MapImg, (self.start[0],self.start[1]), 5, self.Green, self.nodeThickness)
        if nodeType == 'N':
            cv2.circle(self.MapImg, (coords[0], coords[1]), self.nodeRad, self.Blue, self.nodeThickness)
        if nodeType == 'NO':
            cv2.circle(self.MapImg, (coords[0], coords[1]), self.nodeRad, self.Red, self.nodeThickness)
            
    def drawObs(self, obs):
        for i in range(0, len(obs), 2):
            cv2.rectangle(self.MapImg, (obs[i][0], obs[i][1]), (obs[i+1][0], obs[i+1][1]), self.Black, -1)
    
    def drawEdge(self, n1, n2, Normal):
        if Normal:
            cv2.line(self.MapImg, (n1.x, n1.y), (n2.x, n2.y), self.Red, thickness = 1)
            return
        cv2.line(self.MapImg, (n1.x, n1.y), (n2.x, n2.y), self.Green, thickness = 1)

    def isFree(self, n):
        r, g, b = self.dummyMap[n.y,n.x]
        if r+g+b==0:
            return False
        return True
    
    def MapAnimation(self, vertex, goals):
        map_animation = self.dummyMap.copy()
        cv2.circle(map_animation, (vertex[0].x, vertex[0].y), self.nodeRad, self.Blue, self.nodeThickness)
        for i in range(1, len(vertex)):
            cv2.circle(map_animation, (vertex[i].x, vertex[i].y), self.nodeRad, self.Blue, self.nodeThickness)
            cv2.line(map_animation, (vertex[i].parent.x, vertex[i].parent.y), (vertex[i].x, vertex[i].y), self.Red, thickness = 1)
        ####Printing the shortest path if any goal is found
        if len(goals)!=0:
            idx = 0
            mini = vertex[goals[0]].distance
            for i in range(len(goals)):
                if vertex[goals[i]].distance<mini:
                    mini = vertex[goals[i]].distance
                    idx = i
            map_animation = self.drawPath(vertex[goals[idx]], map_animation)
        cv2.imshow(self.MapWindowName, map_animation)
        cv2.waitKey(1)
        
    def drawPath(self, n, map_animation):
        text = str(n.distance)
        curr_n = n
        while(curr_n.parent != None):
            parent = curr_n.parent
            cv2.line(map_animation, (parent.x, parent.y), (curr_n.x, curr_n.y), self.Green, thickness = 2)
            curr_n = curr_n.parent
        cv2.putText(map_animation, text, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)    
        return map_animation

    def crossObstacles(self, n1, n2, goal):
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        upper = 101
        if goal:
            upper = 401
        for i in range(0, upper):
            u = i/(upper-1)
            x = int(x1*u + x2*(1-u))
            y = int(y1*u + y2*(1-u))
            r, g, b = self.dummyMap[y, x]
            if r+b+g==0:
                return True
        return False
        
class RRTGraph:
    
    def __init__(self, nstart, mapdimensions, map):
        (x, y) = nstart
        self.root = Node(x, y)
        self.goal = Node(map.goal[0], map.goal[1])
        self.map = map
        self.obstacles = []
        self.obsDim=40
        self.obsNum=40
        self.mapw, self.maph = mapdimensions[0], mapdimensions[1]
        self.stepSize = 20
        self.goalDim = 1
        self.generateGoalNode = 0
        self.goalNode = []
        self.searchRad = 30

        self.vertex = [self.root]
        self.vertex_index_dict = {self.root:0}
        
    def makeRandomRectangles(self):
        centerx = int(random.uniform(self.obsDim/2, self.mapw-self.obsDim/2))
        centery = int(random.uniform(self.obsDim/2, self.maph-self.obsDim/2))
        return [centerx-int(self.obsDim/2), centery-int(self.obsDim/2)]

    def makeobs(self):
        obs = []
        for i in range(0, self.obsNum-1):
            upper = self.makeRandomRectangles()
            lower = [upper[0]+self.obsDim, upper[1]+self.obsDim]
            while (upper[0]<=self.root.x<=lower[0] and upper[1]<=self.root.y<=lower[1]) or (upper[0]<=self.map.goal[0]<=lower[0] and upper[1]<=self.map.goal[1]<=lower[1]):
                upper = self.makeRandomRectangles()
                lower = [upper[0]+self.obsDim, upper[1]+self.obsDim]
            obs.append(upper)
            obs.append(lower)
        self.obstacles = obs.copy()
        return obs
    
    def metric(self, n1, n2):
        (x1, y1) = (float(n1.x), float(n1.y))
        (x2, y2) = (float(n2.x), float(n2.y))
        return ((x2-x1)**2+(y2-y1)**2)**0.5
    
    def nearest(self, n):
        dmin = self.metric(self.root, n)
        nnear_index = 0
        for i in range(len(self.vertex)):
            ndis = self.metric(self.vertex[i], n)
            if ndis<dmin:
                dmin = ndis
                nnear_index = i
        return nnear_index  
    
    def nearest_in_rad(self, n, nnear):
        neighbour_idx = []
        best = nnear
        dmin = self.metric(n, self.vertex[nnear])
        for i in range(len(self.vertex)):
            ndis = self.metric(self.vertex[i], n)
            if ndis<=self.searchRad:
                neighbour_idx.append(i)
            if ndis<dmin:
                dmin = ndis
                best = i
        return best, neighbour_idx
    
    def step(self, nnear, new_node):
        d = self.metric(nnear, new_node)
        if d>self.stepSize :
            xnear, ynear = nnear.x, nnear.y
            xrand, yrand = new_node.x, new_node.y
            theta = math.atan2(yrand-ynear, xrand-xnear)
            (x, y) = (int(xnear+self.stepSize *math.cos(theta)), int(ynear+self.stepSize*math.sin(theta)))
            new_node.x = x
            new_node.y = y
        return new_node
    
    def isGoal(self, n):
        if self.goal.x == n.x and self.goal.y == n.y:
            return True
        return False

    def drawPath(self, n):
        curr_n = n
        while(curr_n.parent != None):
            parent = curr_n.parent
            self.map.drawEdge(parent, curr_n, False)
            curr_n = curr_n.parent

    def remove_child(self, parent, child):
        for i in range(len(self.vertex[parent].childs)):
            if self.vertex[parent].childs[i] == child:
                self.vertex[parent].childs.pop(i)
                break
    
    def rewrite_distances(self, parent):
        queue = self.vertex[parent].childs
        while(len(queue)!=0):
            temp = queue.pop(0)
            idx = self.vertex_index_dict[temp]
            self.vertex[idx].distance = self.vertex[idx].parent.distance + self.metric(self.vertex[idx], self.vertex[idx].parent)

    def rewire(self, newNode_idx, nneigbours):
        for i in nneigbours:
            if not self.map.crossObstacles(self.vertex[newNode_idx], self.vertex[i], False):
                temp = self.metric(self.vertex[newNode_idx], self.vertex[i])
                if self.vertex[newNode_idx].distance + temp < self.vertex[i].distance:
                    self.vertex[i].distance = self.vertex[newNode_idx].distance + temp
                    parent = self.vertex_index_dict[self.vertex[i].parent]
                    self.remove_child(parent, self.vertex[i])
                    self.vertex[i].parent = self.vertex[newNode_idx]
                    self.rewrite_distances(i)

    def add_node(self, new_node):
        #Checks whether the current node is free and tries to add the node if no obstacels are there in between
        nnear_index = self.nearest(new_node)
        is_Goal = self.isGoal(new_node)
        if not is_Goal:
            new_node = self.step(self.vertex[nnear_index], new_node)
        nbest, nneighbours = self.nearest_in_rad(new_node, nnear_index)
        if not self.map.crossObstacles(self.vertex[nbest], new_node, is_Goal):
            new_node.parent = self.vertex[nbest]
            self.vertex[nbest].childs.append(new_node)
            new_node.distance = self.vertex[nbest].distance + self.metric(self.vertex[nbest], new_node)
            if is_Goal:
                self.goalNode.append(len(self.vertex))
            self.vertex_index_dict[new_node] = len(self.vertex)
            self.vertex.append(new_node)
            self.rewire(len(self.vertex)-1, nneighbours)
            return (self.vertex[nnear_index], new_node, True)

        del new_node
        return (0, 0, False)
    
    def sample_envir(self):
        if self.generateGoalNode == 50:
            x = self.goal.x
            y = self.goal.y
            self.generateGoalNode = 0
            return x, y
        x = int(random.uniform(0, self.mapw))
        y = int(random.uniform(0, self.maph))    
        self.generateGoalNode += 1    
        return x, y
        
map = RRTMap([40, 40], [400,400], [512, 512])
graph = RRTGraph([40, 40], [512, 512], map)
obstacles = graph.makeobs()
map.drawMap(obstacles)
cnt = 0
actual_cnt = 0
while(cnt < 10000):
    x, y = graph.sample_envir()
    new_node = Node(x, y)
    n, new_node, ok = graph.add_node(new_node)
    if ok:
        if graph.isGoal(new_node):
            map.MapAnimation(graph.vertex, graph.goalNode)
        map.MapAnimation(graph.vertex, graph.goalNode)
        cnt += 1
    actual_cnt += 1
cv2.waitKey(0)
