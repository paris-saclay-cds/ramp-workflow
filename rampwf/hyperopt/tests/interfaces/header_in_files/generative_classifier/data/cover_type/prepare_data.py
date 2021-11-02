import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('covtype.csv')



data = data[data['Cover_Type'].isin([1,2])]
data = data.rename(columns={"Cover_Type": "target"})
binary_transf = {1:0, 2:1}
data = data.replace({"target": binary_transf})
train, test = train_test_split(data, stratify = data['target'], train_size=1000)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
