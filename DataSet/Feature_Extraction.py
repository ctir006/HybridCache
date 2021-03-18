#https://www.kaggle.com/nnqkfdjq/statistics-observation-of-random-youtube-video#count_observation_upload.csv

import pandas as pd
from collections import defaultdict
import numpy as np
dataset = pd.read_csv('count_observation_upload.csv')
df = pd.DataFrame(dataset)
# We are insterested in only videos id and view count values so we need 2 and 13 columns in the csv file.
cols = [2,13]
df = df[df.columns[cols]]
df.fillna(0, inplace=True)
d=defaultdict(list)
for i, row in enumerate(df.values[1:]):
	d[row[0]].append(row[1])
for i,key in enumerate(d):
	if i!=0:
		d[key]=d[key][1:]
features=[]
for i in d.values():
	features.append(i)
features=np.array(features)

# Normalizing view count values
features=features/features.sum(axis=0)
features[features < 0] = 0
for i in features:
	print(i)

# Shapre of features matrix is 1611 X 694. Which means 1611 videos over 694 time series values.
print(len(features),len(features[0]))


	

