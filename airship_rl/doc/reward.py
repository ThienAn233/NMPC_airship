import matplotlib.pyplot as plt
import numpy as np
n   = 50
xs  = np.linspace(0,1,n)
ys  = np.linspace(0,1,n)
zs  = np.linspace(-1,1,n)
X,Y,Z = np.meshgrid(xs,ys,zs)

def reward(location,target):
    xm, ym, zm = location
    xt, yt, zt = target
    l = np.linalg.norm(np.linalg.cross(location,target))/np.linalg.norm(target)
    da = np.linalg.norm(location-target)
    const = np.linalg.norm(target)
    return -20*l - 10*(np.abs(da) - const)
target = np.array([1,1,1])
ws = []
for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten()):
    ws.append(reward([x,y,z],target))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X.flatten(),Y.flatten(),Z.flatten(),c=ws)
plt.colorbar(ax.scatter(X.flatten(),Y.flatten(),Z.flatten(),c=ws))
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_zscale('linear')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()