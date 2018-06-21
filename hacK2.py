import numpy as np
import tensorflow as tf

feature_columns = [tf.feature_column.numeric_column(
    "x", shape=[1]
)]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
x_train = np.array([1.0, 2.0, 3.0, 6.0, 8.0])
y_train = np.array([4.8, 8.5, 10.4, 21.0, 25.3])

x_eval = np.array([2.0, 5.0, 7.0, 9.0])
y_eval = np.array([7.6, 17.2, 23.6, 28.8])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x":x_train}, y_train, batch_size=2, num_epochs=1000, shuffle=True
)
train_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
    {"x":x_train}, y_train, batch_size=2, num_epochs=1000,shuffle=False
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x":x_eval},y_eval, batch_size=2,num_epochs=1000, shuffle=False
)

estimator.train(input_fn=train_input_fn, steps=1000)
train_metrics = estimator.evaluate(
    input_fn=train_input_fn_2
)
print(estimator.params)
print("train metrics:%r"%train_metrics)
eval_metrics = estimator.evaluate(
    input_fn=eval_input_fn
)
print("eval metrics:%s"%eval_metrics)
