import wget
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import json

url = 'https://www.openml.org/data/get_csv/183151/php7J0u2S'
target_file = 'click.csv'
if os.path.exists(target_file):
    print("already downloaded")
else:
    print("downloading")
    wget.download(url, target_file)
print("reading data")
data = pd.read_csv(target_file,error_bad_lines=False)
if not os.path.exists(target_file) :
    print("to data file")
    data.to_csv(target_file, index=False)
data = data.rename(columns={"click":"target"})
print("data is", data)
n_values = data.nunique()

cat = n_values[n_values < 50]
num = n_values[n_values/len(data) > 50]
print(cat)
print(len(n_values),"length")
#types = dict()
#for i in cat.index:
 #   if i!= "target" : 
  #      types[i] = "cat"
#for i in num.index:
 #   types[i] = "num"
#with open("dtypes.json", "w") as out:
 #   json.dump(types, out)


train, test = train_test_split(data, train_size = 1000)
train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
