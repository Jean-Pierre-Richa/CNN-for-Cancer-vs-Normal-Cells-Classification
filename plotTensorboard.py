# plot data tensorboard
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# root folder
root_dir = os.getcwd()
# visualization folder
visualization_dir = os.path.join(root_dir, "visualization_tensorboard/")

train_accuracy_file = os.path.join(visualization_dir, "final-train-accuracy.csv")
train_loss_file = os.path.join(visualization_dir, "final-train-loss.csv")
test_accuracy_file = os.path.join(visualization_dir, "final-test-accuracy.csv")
test_loss_file = os.path.join(visualization_dir, "final-test-loss.csv")

def graph(csv_file):
    x_=[]
    y_=[]
    with open(csv_file, 'r') as tab_file:
        next(tab_file)
        reader = csv.reader(tab_file, delimiter=',')
        for _, value1, value2 in reader:
            x_.append(int(value1))
            y_.append(float(value2))

    f = interp1d(x_, y_)
    f2 = interp1d(x_, y_, kind='cubic')
    xnew = np.linspace(0, 14, num=100, endpoint=True)
    plt.plot(xnew, f2(xnew), 'bo-', markevery=7)
    plt.title(str(csv_file.split("-")[-2]).capitalize()[0:] + "_" + \
                                str(csv_file.split("-")[-1].split(".")[0]))
    plt.ylabel(csv_file.split("-")[-1].split(".")[0].capitalize()[0:])
    plt.xlabel('Epoch')
    plt.show()

graph(train_accuracy_file)
graph(train_loss_file)
graph(test_accuracy_file)
graph(test_loss_file)
