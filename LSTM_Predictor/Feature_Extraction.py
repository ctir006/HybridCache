#https://www.kaggle.com/nnqkfdjq/statistics-observation-of-random-youtube-video#count_observation_upload.csv

import pandas as pd
from collections import defaultdict
import numpy as np


class YouTube:
	def __init__(self,filename):
		self.dataset = pd.read_csv(filename)
	
	def get_features(self):
		df 	= pd.DataFrame(self.dataset)
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
		np.seterr(divide='ignore', invalid='ignore')
		#features=features/features.sum(axis=0) 
		features[features < 0] = 1
		#features=np.around(features, decimals=4)
		where_are_NaNs = np.isnan(features)
		features[where_are_NaNs] = 1
		return features

	
	
#[ 5881096. 15129395.  4786621.  8860295.  8254700. 11275645. 12121089. 12137170. 13875160. 12400016.]
#features=np.around(features, decimals=5)