import pandas as pd
import numpy as np

from random import random
import random
import seaborn as sns
from google.colab import files
size=10000

#sound
fta1=np.random.exponential(2.38,size)
#noise some reduntant features

fta2=np.random.uniform(82,90,size)
ftb2=np.random.uniform(82,90,size)
#lat and lon
west_lat = np.random.uniform(81, 91, size)
west_long = np.random.uniform(35, 42, size)

# Specify weight.
west_weight = np.random.uniform(9, 19, size)
west_weight = np.random.normal(3, 1.25, size) + west_weight

#add noise for species A
idx = random.sample(range(0, len(west_weight)), 2000)

for i in idx:
  west_weight[i]=np.nan
# Specify wing-span (which has a quadratic relationship with weight).
west_wing = np.random.normal(5.979, 1, size) + 1.871 * \
            west_weight + -.0396 * west_weight**2

# Specify the feather colors.
west_color = pd.Series(np.random.randint(0, 2, 
                                         size)).map({0:'Black', 
                                                     1:'Yellow'}).astype('category')
#add noise for species A color - setting them empty                                      
for i in idx:
  west_color[i]=np.nan

# Specify the beak length.
west_beak = np.random.uniform(0.1, 15, size)
west_beak = np.random.normal(0, 1.25, size) + west_weight


#Eastern variety Latitude & Longitude with half the
ftb1=np.random.exponential(2.38,size)
east_lat = np.random.uniform(77, 89, size)
east_long = np.random.uniform(34, 39, size)

# Specify weight.
east_weight = np.random.uniform(10.5, 24.5, size)
east_weight = np.random.normal(0, .66, size) + east_weight
#add noise for species B
idx_B = random.sample(range(0, len(west_weight)), 1000)

for i in idx_B:
  east_weight[i]=np.nan
  east_lat[i]=west_lat[i]
# Specify wing-span (which has a quadratic relationship with weight).
east_wing = np.random.normal(24.16, .75,size) + -.137 * \
            east_weight + .0119 * east_weight**2

# Specify the feather colors.
east_color = pd.Series(np.random.randint(0, 3,
                                         size)).map({0:'Black', 
                                                     1:'Yellow', 
                                                     2:'White'}).astype('category')

#add noise for species B color
for i in idx_B:
  east_color[i]=np.nan


# Specify the beak length.
east_beak = np.random.uniform(5, 34, size)
east_beak = np.random.normal(3, 2.5, size) + west_weight


#add useless dataset: weather data.
times1 = pd.date_range(end='1/1/2021', periods=size, name="time")
times2 = pd.date_range(end='1/1/2021', periods=size, name="time")
annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))

base = 10 + 15 * annual_cycle
tmin_valuesEast = base + np.random.randn(annual_cycle.size)
tmax_valuesEast = base + 10 + np.random.randn(annual_cycle.size)

tmin_valuesWest = base + np.random.randn(annual_cycle.size)
tmax_valuesWest = base + 10 + np.random.randn(annual_cycle.size)

west = pd.DataFrame({'variety':[1] * size,'sound':fta1,'s':fta1,'number':fta2,'lat':west_lat,'l':east_lat,'long':west_long,
                     'weight':west_weight, 'wing':west_wing,
                     'color':west_color,
                     'time':times1,'temp_min':tmin_valuesWest, 'temp_max':tmax_valuesWest,
                     'beak':west_beak})
east = pd.DataFrame({'variety':[0] * size,'sound':ftb1,'s':ftb1,'number':ftb2,'lat':east_lat,'l':west_lat,'long':east_long,
                     'weight':east_weight, 'wing':east_wing,
                     'color':east_color,
                     'time':times2,'temp_min':tmin_valuesEast, 'temp_max':tmax_valuesEast, 
                     'beak':east_beak})
df = pd.concat([west, east])
mapping = {'Yellow': 1,'Black': 2,'White':3}
df_1=df.applymap(lambda s: mapping.get(s) if s in mapping else s)
df_1


#add noise for species A and B color column
df_1.to_csv('MyData_1.csv', index=False)
files.download('MyData_1.csv')
#add noise for species A and B color column
# df_1.to_csv('MyData_4_test.csv', index=False)
# files.download('MyData_2.csv')