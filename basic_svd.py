import graphlab
from graphlab.toolkits.recommender.factorization_recommender import FactorizationRecommender
import pandas as pd

from utils import average_precision_at_k

train = graphlab.SFrame.read_csv("eval_test.csv")
test = graphlab.SFrame.read_csv("train.csv")
model = graphlab.recommender.ranking_factorization_recommender.create(train,
                                                                      user_id='node1',
                                                                      item_id='node2',
                                                                      target='label',
                                                                      num_factors=10)
predictions = model.predict(test)

df = pd.read_csv('train.csv')
df['score'] = list(predictions)
df = df.sort_values(by=['score'], ascending=False)
# df["node1_node2"] = df["node1"].map(str) + "_"+ df["node2"].map(str)
# df.to_csv('for_kaggle.csv', columns = ['node1_node2'], index=False)
class_correct = list(df['label'])

print "\n\n the precision at k score is {}".format(average_precision_at_k(100, class_correct))

pass


pass