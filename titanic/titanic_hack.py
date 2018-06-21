import tensorflow as tf
import numpy as np
import pandas as pd

from titanic import titanic_data

def main(argv):
    (train_x, test_x),(train_y, test_y) = titanic_data.load_data()
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10,10],
        n_classes=2
    )
    classifier.train(
        input_fn=lambda :titanic_data.train_input_fn(train_x, train_y,100),
        steps=1000
    )

    eval_train = classifier.evaluate(
        input_fn=lambda : titanic_data.eval_input_fn(train_x,train_y,100)
    )
    eval_result = classifier.evaluate(
        input_fn=lambda: titanic_data.eval_input_fn(test_x, test_y, 100)
    )
    print(eval_train)
    print(eval_result)
    prediction = classifier.predict(input_fn=lambda:titanic_data.eval_input_fn(test_x, labels=None, batch_size=100))

    expected = test_y
    cnt = 0
    cor = 0
    for pred_dict, expec in zip(prediction,expected):
        cnt += 1
        classid = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][classid]
        if int(classid) == int(expec):
            cor += 1
        print('predict result:%s, prob:%.3f, expect:%s'%(classid, probability, expec))
    print('my accuracy:%s'%(cor/cnt))




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

