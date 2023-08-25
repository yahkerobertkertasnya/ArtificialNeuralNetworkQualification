import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# atau

#import tensorflow as tf


def data_reader():
    data = pd.read_csv('SinX+CosY.csv')
    input_data = data.iloc[:, :-1]
    output_data = data.iloc[:, -1:]


    input_data_scaled = input_scaler.fit_transform(input_data)

    return input_data_scaled, output_data


def feed_forward(datasets):

    input_to_hidden = tf.matmul(datasets, input_hidden["weight"]) + input_hidden["bias"]
    hidden_activation = tf.nn.tanh(input_to_hidden)

    hidden_to_output = tf.matmul(hidden_activation, hidden_output["weight"]) + hidden_output["bias"]

    return hidden_to_output


if __name__ == '__main__':
    print("Training neural network for sin(x) + cos(y)...")
    input_scaler = MinMaxScaler()

    input_data, output_data = data_reader()

    layer = {
        "input": 2,
        "hidden": 8,
        "output": 1
    }

    input_hidden = {
        "weight": tf.Variable(tf.random.normal([layer["input"], layer["hidden"]])),
        "bias": tf.Variable(tf.random.normal([layer["hidden"]]))
    }

    hidden_output = {
        "weight": tf.Variable(tf.random.normal([layer["hidden"], layer["output"]])),
        "bias": tf.Variable(tf.random.normal([layer["output"]]))
    }

    input_values = tf.placeholder(tf.float32, [None, layer["input"]])
    target_values = tf.placeholder(tf.float32, [None, layer["output"]])

    output = feed_forward(input_values)

    mse_loss = tf.reduce_mean(tf.square(target_values - output))
    train_step = tf.train.AdamOptimizer(0.05).minimize(mse_loss)

    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.4)

    epoch = 2000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch + 1):
            train_dict = {
                input_values: x_train,
                target_values: y_train
            }

            sess.run(train_step, feed_dict=train_dict)

            loss = sess.run(mse_loss, feed_dict=train_dict)

            if i % 500 == 0:
                print(f'Iteration: {i}, Current MSE Loss: {loss}')

        test_dict = {
            input_values: x_test,
            target_values: y_test
        }

        test_loss = sess.run(mse_loss, feed_dict=test_dict)
        print(f'Test MSE Loss: {test_loss}')

        while True:
            x_value = input("Input a number for x ")
            y_value = input("Input a number for y ")

            input_dataframe = pd.DataFrame(np.array([[x_value, y_value]]), columns=['x', 'y'])
            x_test = input_scaler.transform(input_dataframe)
            y_test = sess.run(output, feed_dict={input_values: x_test})
            print('Predicted Output:', y_test[0][0])
