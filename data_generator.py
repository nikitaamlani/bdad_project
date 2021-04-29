# Standard Imports
import pandas as pd
import numpy as np

from random import random

import seaborn as sns
%matplotlib inline
# Western variety Latitude & Longitude.
west_lat = np.random.uniform(81, 91, 1000)
west_long = np.random.uniform(35, 42, 1000)

# Specify weight.
west_weight = np.random.uniform(9, 19, 1000)
west_weight = np.random.normal(0, 1.25, 1000) + west_weight

# Specify wing-span (which has a quadratic relationship with weight).
west_wing = np.random.normal(5.979, 1, 1000) + 1.871 * \
            west_weight + -.0396 * west_weight**2

# Specify the feather colors.
west_color = pd.Series(np.random.randint(0, 2, 
                                         1000)).map({0:'Black', 
                                                     1:'Yellow'}).astype('category')

# Eastern variety Latitude & Longitude.
east_lat = np.random.uniform(77, 89, 1000)
east_long = np.random.uniform(34, 39, 1000)

# Specify weight.
east_weight = np.random.uniform(10.5, 24.5, 1000)
east_weight = np.random.normal(0, .66, 1000) + east_weight

# Specify wing-span (which has a quadratic relationship with weight).
east_wing = np.random.normal(24.16, .75, 1000) + -.137 * \
            east_weight + .0119 * east_weight**2

# Specify the feather colors.
east_color = pd.Series(np.random.randint(0, 3, 
                                         1000)).map({0:'Black', 
                                                     1:'Yellow', 
                                                     2:'White'}).astype('category')
west = pd.DataFrame({'label':west_wing*west_long,'lat':west_lat,'long':west_long,
                     'weight':west_weight, 'wing':west_wing,
                     'color':west_color,'variety':['Western'] * 1000})
east = pd.DataFrame({'label':east_wing*east_long,'lat':east_lat,'long':east_long,
                     'weight':east_weight, 'wing':east_wing,
                     'color':east_color,'variety':['Eastern'] * 1000})
df = pd.concat([west, east])
df.to_csv('BirdVarietyData.csv', index=False)






# Standard Imports
import pandas as pd
import numpy as np

from random import random

import seaborn as sns
%matplotlib inline


fta1=np.random.exponential(2.38,1000)
fta2=np.random.normal(10.4,1.25, 1000)
fta3=np.random.uniform(35, 42, 1000)
fta4=np.random.uniform(27, 116, 1000)
fta5=np.random.normal(0, 1.25, 1000)
fta6=np.random.normal(5.979, 1, 1000)+np.random.uniform(73, 106, 1000)


ftb1=np.random.exponential(3.45,1000)
ftb2=np.random.normal(1.3, 1, 1000)
ftb3=np.random.uniform(12, 32, 1000)
ftb4=np.random.uniform(3, 56, 1000)
ftb5=np.random.normal(5, 1.25, 1000)
ftb6=np.random.normal(2.979, 1, 1000)+np.random.uniform(73, 106, 1000)


# def normalization(x):

#   normalized = (x-min(x))/(max(x)-min(x))

#   return normalized


data_a=pd.DataFrame('label':0,'ft1'=fta1,'ft2'=fta2,'ft3'=fta3,'ft4'=fta4,'ft5'=fta5,'ft6'=fta6)
data_b=pd.DataFrame('label':1,'ft1'=ftb1,'ft2'=ftb2,'ft3'=ftb3,'ft4'=ftb4,'ft5'=ftb5,'ft6'=ftb6)
df=pd.concat([data_a,data_b])
df.to_csv('MyData.csv', index=False)

# west = pd.DataFrame({'label':west_wing*west_long,'lat':west_lat,'long':west_long,
#                      'weight':west_weight, 'wing':west_wing,
#                      'color':west_color,'variety':['Western'] * 1000})
# east = pd.DataFrame({'label':east_wing*east_long,'lat':east_lat,'long':east_long,
#                      'weight':east_weight, 'wing':east_wing,
#                      'color':east_color,'variety':['Eastern'] * 1000})
# df = pd.concat([west, east])
# df.to_csv('BirdVarietyData.csv', index=False)