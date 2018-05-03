import graphlab
from graphlab.toolkits.recommender.factorization_recommender import FactorizationRecommender
import pandas as pd


train = graphlab.SFrame.read_csv("train_train.csv")
test = graphlab.SFrame.read_csv("train_test.csv")
model = graphlab.recommender.ranking_factorization_recommender.create(train, user_id='node1', item_id='node2',target='label')
a = model.predict(test)

df = pd.read_csv('train_test.csv')
df['score'] = list(a)
df = df.sort_values(by=['score'], ascending=False)



pass