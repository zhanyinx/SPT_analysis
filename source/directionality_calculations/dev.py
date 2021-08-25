import pandas as pd
import numpy as np
import time

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate_track(n=10000):
    x = [np.random.uniform(0, 100)]
    y = [np.random.uniform(0, 100)]
    z = [np.random.uniform(0, 100)]
    for i in range(n - 1):
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        x.append(x[-1] + np.sin(theta) * np.cos(phi))
        y.append(y[-1] + np.sin(theta) * np.sin(phi))
        z.append(z[-1] + np.cos(theta))
    return x, y, z
        
single_traj = pd.DataFrame()
track = generate_track(n=10000)
single_traj['x'] = track[0]
single_traj['y'] = track[1]
single_traj['z'] = track[2]
single_traj['frame'] = np.arange(10000)
to_drop = list(np.random.randint(0,10000,7))
single_traj = single_traj.drop(to_drop)
single_traj = single_traj.reindex()

def angle_between_vectors(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0 or np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
        return np.nan
    return np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
dt = 10
distr = []
# loop over single traj
init = time.time()
for t in single_traj.frame.values:
    if (np.count_nonzero(single_traj.frame.values == t) == 1):
        i = single_traj.index[single_traj.frame == t][0]
    else:
        continue
    if (np.count_nonzero(single_traj.frame.values == t + dt) == 1):
        j = single_traj.index[single_traj.frame == t + dt][0]
    else:
        continue
    if (np.count_nonzero(single_traj.frame.values == t + 2*dt) == 1):
        k = single_traj.index[single_traj.frame == t + 2*dt][0]
    else:
        continue

    v1 = [
        single_traj.loc[j].x - single_traj.loc[i].x,
        single_traj.loc[j].y - single_traj.loc[i].y,
        single_traj.loc[j].z - single_traj.loc[i].z,
        ]
    v2 = [
        single_traj.loc[k].x - single_traj.loc[j].x,
        single_traj.loc[k].y - single_traj.loc[j].y,
        single_traj.loc[k].z - single_traj.loc[j].z,
        ]
    distr.append(angle_between(v1, v2)/np.pi*180)
print ("stackoverflow ", time.time() - init)

import matplotlib.pyplot as plt

plt.hist(distr, bins=180)
plt.show()
plt.clf()

plt.hist(single_traj['x'].values, bins=100)
plt.show()
plt.clf()

plt.hist(single_traj['y'].values, bins=100)
plt.show()
plt.clf()

plt.hist(single_traj['z'].values, bins=100)
plt.show()
plt.clf()