import random
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import distance

#parameters:

epsilon = 1
minPts = 20

cluster = []
core_points = []

#exaple data

def GenerateData():
    
    x1=numpy.random.randn(50,2)
    x2x=numpy.random.randn(80,1)+12
    x2y=numpy.random.randn(80,1)
    x2=numpy.column_stack((x2x,x2y))
    x3=numpy.random.randn(100,2)+8
    x4=numpy.random.randn(120,2)+15
    data=numpy.concatenate((x1,x2,x3,x4))
    data = numpy.ndarray.tolist(data)
    
    return data

data = GenerateData()
# print(z)
# plt.plot(data, '.')

def RandomPointPick(data):
    random_start_point = random.choice(data)
    return random_start_point

random_start_point = RandomPointPick(data)

def CheckEpsilon(data, random_start_point, epsilon):
        for i in data:
            if distance.euclidean(i, random_start_point) <= epsilon:
                core_points.append(i)  
                core_points.append(random_start_point)
                # print(data)
                # print(core_points)
                [item for item in data if item not in core_points]
                if len(core_points) >= minPts:
                    for j in data:
                        for k in core_points:
                            if distance.euclidean(j, k) <= epsilon:
                                cluster.append(k)
                    full_claster = (*cluster, *core_points)
                    return data,  full_claster
    
# def CheckMinPts(core_points):
#     if len(core_points) >= minPts:
#         print("Is cluster")
#         for i in core_points:
#             if 
#     else:
#         print("No claster")
        
        

core_points, full_claster = CheckEpsilon(data, random_start_point, epsilon)
plt.plot(full_claster, '.')
# CheckMinPts(core_points)