import iris_func
import tensorflow as tf
import numpy as np

feature_columns = [tf.contrib.layers.real_valued_column('', dimension = 4)]

dnn_clf = tf.contrib.learn.DNNClassifier(
    feature_columns = feature_columns,
    hidden_units = [10],
    n_classes = 3,
    model_dir = './iris_tfcontrib'
)

x_train, x_test, y_train, y_test = iris_func.get_iris_data()
#print(iris_func.get_iris_data())

def input_fn_train():
    x = tf.constant(x_train)
    y = tf.constant(y_train)
    return x, y

dnn_clf.fit(input_fn = input_fn_train, steps = 1000)

def input_fn_test():
    x = tf.constant(x_test)
    y = tf.constant(y_test)
    return x, y

accuracy = dnn_clf.evaluate(input_fn = input_fn_test, steps = 1)['accuracy']
print('accuracy: ', accuracy)

def input_fn_predict():
    predict = np.array([[4.9, 3.1, 1.5, 0.3], [7.2, 3.0, 4.5, 1.5]], dtype = np.float32)
    return predict

predict_list = list(dnn_clf.predict_classes(input_fn = input_fn_predict))
#print(predict_list)

for predict in predict_list:
    label = ''
    if predict == 0:
        label = 'setosa'
    elif predict == 1:
        label = 'versicolor'
    else:
        label = 'virginica'
    print(label)

