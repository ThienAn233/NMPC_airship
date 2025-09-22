import matplotlib.pyplot as plt
import numpy as np
# 2D case
n   = 100
xs  = np.linspace(0,2,n)
ys  = np.linspace(0,2,n)
X,Y = np.meshgrid(xs,ys)
def reward_2d(location,target):
    xm, ym = location
    xt, yt = target
    l = np.linalg.norm(np.cross(location,target))/np.linalg.norm(target)
    da = np.linalg.norm(location-target)
    const = np.linalg.norm(target)
    return -20*l - 2*(np.abs(da) - const)
target = np.array([1,1])
ws = [] 
for x,y in zip(X.flatten(),Y.flatten()):
    ws.append(np.exp(reward_2d(np.array([x,y]),target)))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.array(ws).reshape(n,n), cmap='viridis')
plane= ax.plot_surface(X, Y, np.zeros((n,n)), alpha=0.2)
ax.set_xlim(-0,2)
ax.set_ylim(-0,2)
ax.scatter3D(target[0],target[1],reward_2d(target,target)+0.5,c='r')
plt.colorbar(surf)
plt.show()