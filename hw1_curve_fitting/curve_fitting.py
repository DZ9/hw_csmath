import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_polynomial_param(x, n):
    X = np.tile(x, (1, (n + 1)))
    for i in range(n + 1):
        X[:, i] = X[:, i] ** i
        # params.append(tmp)
    # return params
    return X

def inference():
    pass

def input_fn(x):
    x = x[:, np.newaxis]
    y = np.sin(x)

    return x, y

def noise(y):
    y += np.random.normal(0.1, 0.1, y.shape)
    return y

def poly_interpolation(degree, x_input, y_input, x_test, trainning_step=200000):
    x = tf.placeholder(tf.float32, [None, degree + 1], name='input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='label')

    w = tf.Variable(tf.truncated_normal([degree + 1, 1], stddev=0.1))
    y = tf.matmul(x, w)

    loss = tf.reduce_sum((y - y_) ** 2)

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model_input = generate_polynomial_param(x_input, degree)

    tmp = 0
    step = 0
    flag = False
    for i in range(trainning_step):
        train_step.run(feed_dict={x: model_input, y_: y_input})
        le = loss.eval(feed_dict={x: model_input, y_: y_input})
        if le - tmp == 0 and not flag:
            flag = True
        if le - tmp == 0 and flag:
            flag = False
            step = i
            break
        if le - tmp != 0 and flag:
            flag = False
        tmp = le


    le = loss.eval(feed_dict={x: model_input, y_: y_input})
    print 'trainning loss is ', le, 'on degree ', degree, 'after ', step, ' trainning steps'

    test_model_input = generate_polynomial_param(x_test, degree)
    y_output = y.eval(feed_dict={x: test_model_input})

    return y_output



if __name__ == '__main__':
    x = np.linspace(-5, 5, 10)
    x_label, y_label = input_fn(x)
    y_label = noise(y_label)

    x_test = np.linspace(-90, 90, 900)
    x_test_label, y_test_label = input_fn(x_test)

    labels = []
    for i in range(10):
        if i == 0:
            continue
        labels.append(poly_interpolation(i, x_label, y_label, x_test_label))

    plt.figure()
    begin = 331

    for i in range(len(labels)):
        plt_tmp = plt.subplot(begin + i)
        plt_tmp.plot(x_test_label, labels[i], color='red', label="$polynomial$")
        plt_tmp.axis([-10.0, 10.0, -2.0, 2.0])
        plt_tmp.plot(x_label, y_label, color="blue", linestyle='None', marker='o')
        plt_tmp.plot(x_test_label, y_test_label, color="purple", label="$sinx", linestyle='dashed')
        plt_tmp.set_title('Degree=%d'%(i+1))
        plt_tmp.legend()

    plt.show()

    '''
    plt.xlim((-10, 10))
    plt.ylim((-2, 2))
    plt.plot(x_value, y_input, color="blue", linestyle='None', marker='o')
    plt.plot(x_test_value, y_output, color="red", )
    plt.plot(x_test_value, y_sin, color="purple", )
    plt.show()
    '''