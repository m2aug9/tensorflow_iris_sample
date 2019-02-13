from sklearn.model_selection import train_test_split
import numpy as np

def get_iris_data():
    x_data = []
    y_data = []
    f = open('./iris.csv','r')
    line = f.readline()
    while line:
        line_param = line.rstrip().split(',')
        if len(line_param) == 5:
            input = line_param[:len(line_param)-1]
            label = line_param[len(line_param)-1]
            if label == 'setosa':
                label = 0
            elif label == 'versicolor':
                label = 1
            else:
                label = 2
            x_data.append(input)
            y_data.append(label)
        line = f.readline()
    f.close()

    return train_test_split(
        np.array(x_data, dtype = np.float32),
        np.array(y_data, dtype = np.int32),
        test_size = 0.3
    )
