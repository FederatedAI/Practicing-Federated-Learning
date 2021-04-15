
from sklearn.datasets import load_boston
import pandas as pd 


boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

#boston = boston.dropna()
boston = (boston-boston.mean())/(boston.std()) 

col_names = boston.columns.values.tolist()



columns = {}
for idx, n in enumerate(col_names):
	columns[n] = "x%d"%idx 

boston = boston.rename(columns=columns)	

boston['y'] = boston_dataset.target

boston['idx'] = range(boston.shape[0])

idx = boston['idx']

boston.drop(labels=['idx'], axis=1, inplace = True)

boston.insert(0, 'idx', idx)


eval = boston.iloc[406:]

df1 = eval.sample(80)
df2 = eval.sample(85)

housing_1_eval = df1[["idx", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]]

housing_1_eval.to_csv('housing_1_eval.csv', index=True, header=True)


housing_2_eval = df2[["idx", "y", "x8", "x9", "x10", "x11", "x12"]]

housing_2_eval.to_csv('housing_2_eval.csv', index=True, header=True)
