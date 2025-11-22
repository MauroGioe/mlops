import pandas as pd


data = pd.read_csv("./data/creditcard.csv")

data = data.drop(["Time"],axis=1)
train=data.sample(frac=0.25,random_state=200)
valid_test_prod=data.drop(train.index)
valid = valid_test_prod.sample(frac=0.333,random_state=200)
test_prod=valid_test_prod.drop(valid.index)
test = test_prod.sample(frac=0.50,random_state=200)
prod=test_prod.drop(test.index)



train.to_csv("./data/train.csv")
valid.to_csv("./data/valid.csv")
test.to_csv("./data/test.csv")
prod.to_csv("./data/prod.csv")