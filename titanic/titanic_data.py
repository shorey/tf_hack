import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

csv_columns_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
label = [0, 1]
train_file = 'titanic/train.csv'
test_file = 'titanic/test.csv'


def load_data(label ='Survived'):
    train_df = pd.read_csv(train_file, header=0)[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X, y = train_df, train_df.pop(label)
    X.fillna(-1,inplace=True)
    X.loc[:,'Sex'] = X['Sex'].apply(lambda x: 1 if x=='male'  else 0)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,random_state=0)

    return (train_x, test_x),(train_y,test_y)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

if __name__ == '__main__':
    (train_x, test_x), (train_y, test_y) = load_data()

