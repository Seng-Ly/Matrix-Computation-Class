
import numpy as np
import matplotlib.pyplot as plt

data    = np.genfromtxt("data-nonlinear.txt", delimiter=',')

pointX  = data[:, 0]
pointY  = data[:, 1]
label   = data[:, 2]

pointX0 = pointX[label == 0]
pointY0 = pointY[label == 0]

pointX1 = pointX[label == 1]
pointY1 = pointY[label == 1]

x_data = pointX
y_data = pointY
l_data = label

# plt.figure(figsize=(8, 8))
# plt.scatter(pointX0, pointY0, c='b')
# plt.scatter(pointX1, pointY1, c='r')
# plt.tight_layout()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

def g(x,y,theta):
    sum = 0
    for i in range(len(theta)):
        for j in range(len(theta[0])):
            sum += theta[i][j]*(x**i)*(y**j)
    return sum

def dg_dtheta(x,y,i,j):
    return np.abs ((x**i) * (y**j))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def j1(x,y,l,theta):
    total = 0
    if l == 1:
        total = - np.log(sigmoid(g(x,y,theta)))
    if l == 0:
        total = - np.log(1 - sigmoid(g(x,y,theta)))
    return total

def sumtheta2(theta):
    sum = 0
    for i in range(len(theta)):
        for j in range(len(theta[0])):
            sum += theta[i][j]**2
    return sum


theta = np.ones([10,10]).tolist()

m = len(x_data)
alpha = 0.1
lbda = 0.1

# print(sigmoid(g(x_data[0],y_data[0],theta)) - l_data[0])
# print(dg_dtheta(x_data[0],y_data[0],1,2))

for iteration in range(1000):

    for ii in range(len(theta)):
        for jj in range(len(theta[0])):
            sum1m = 0
            for i in range(m):
                sum1m += (sigmoid(g(x_data[i],y_data[i],theta)) - l_data[i]) #*dg_dtheta(x_data[i],y_data[i],ii,jj)
            total = sum1m + lbda*theta[ii][jj]
            theta[ii][jj] = theta[ii][jj] - alpha*total/m

    # print(alpha*total/m)

    sumj = 0
    for i in range(m):
        sumj += j1(x_data[i],y_data[i],l_data[i],theta)
    print(iteration, ((sumj + sumtheta2(theta)))/m)

print (np.array(theta))

