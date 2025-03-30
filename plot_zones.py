import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from precompute_farm_layouts import load_farm_boundaries

plt.figure(figsize=(10, 10))
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
zones = load_farm_boundaries()
for zz, zone_coords in enumerate(zones):
    plt.plot(zone_coords[0], zone_coords[1])
    plt.fill(zone_coords[0], zone_coords[1], alpha=0.2, label=zz)

SIZE = 2

with h5py.File('re_precomputed_layouts.h5', 'r') as f:
   for key in f.keys():
       if 'farm' not in key: continue
       farm = int(key.split('_')[0].strip('farm'))
       seed = int(key.split('_')[2].strip('s'))
       turbine = int(key.split('_')[1][1])
       if farm == 0 and seed > 0: continue
       x, y = f[key]['layout']
       #if turbine != 5: continue
       if farm == 0 and seed == 0 and turbine == 5:
           plt.scatter(x, y, c='k', alpha=1, s=SIZE)
       else:
           plt.scatter(x, y, c='k', alpha=0.003, s=SIZE)
plt.legend()
plt.savefig('zones_updated')
plt.clf()


