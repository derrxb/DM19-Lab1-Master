import pandas as pd
import helpers.data_mining_helpers as dmh
from sklearn.datasets import fetch_20newsgroups


categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42)

# Answer here
# for t in twenty_train.data[:3]:
#     print(t)

X = pd.DataFrame.from_records(dmh.format_rows(twenty_train), columns=['text'])

X['category'] = twenty_train.target
X['category_name'] = X.category.apply(
    lambda t: dmh.format_labels(t, twenty_train))

print(X[0:10])
