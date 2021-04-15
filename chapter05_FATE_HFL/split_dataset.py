
from sklearn.datasets import load_breast_cancer
import pandas as pd 


breast_dataset = load_breast_cancer()

breast = pd.DataFrame(breast_dataset.data, columns=breast_dataset.feature_names)

breast = (breast-breast.mean())/(breast.std()) 

col_names = breast.columns.values.tolist()



columns = {}
for idx, n in enumerate(col_names):
	columns[n] = "x%d"%idx 

breast = breast.rename(columns=columns)	

breast['y'] = breast_dataset.target

breast['idx'] = range(breast.shape[0])

idx = breast['idx']

breast.drop(labels=['idx'], axis=1, inplace = True)

breast.insert(0, 'idx', idx)

breast = breast.sample(frac=1)

train = breast.iloc[:469]

eval = breast.iloc[469:]

breast_1_train = train.iloc[:200]



breast_1_train.to_csv('breast_1_train.csv', index=False, header=True)


breast_2_train = train.iloc[200:]

breast_2_train.to_csv('breast_2_train.csv', index=False, header=True)

eval.to_csv('breast_eval.csv', index=False, header=True)



