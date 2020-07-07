import os
import argparse
import config
import numpy as np
from utils import softmax, onehot_encode, cross_entropy, to_classlabel
from data_loader import get_vector, get_label, data_loader
### implement softmax regression

def train_model(xtrain, ytrain, args):
    # simply randomly initialze W
    iters = args.iterations
    lr = args.lr 
    lamb = args.lamb

    features = xtrain.shape[1]
    class_num = ytrain.shape[1]

    # randomly initialize parameter W
    W = np.random.rand(features+1, class_num)

    for epoch in range(iters):
        running_loss = 0
        for i, data in enumerate(data_loader(xtrain, ytrain, args.batch_size),0):
            x, y = data
            b = np.ones((x.shape[0], 1))
            x = np.append(x, b, 1)
            y_ = softmax(np.dot(x, W))
            W += (np.dot(x.T, y-y_) + lamb*W)/x.shape[0]*lr;
            running_loss += cross_entropy(y, y_)
            if i % 1000 == 999:    # print every 2000 batches
                print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    return W

def test_accuracy(W, xtest, ytest):
    accuracy = 0.0
    test_size = xtest.shape[0]
    xb = np.concatenate((xtest, np.ones(((xtest.shape[0]), 1))), axis=1)
    output = softmax(np.dot(xb, W))
    predicted = to_classlabel(output) # 
    accuracy = np.mean(predicted == ytest) 
    return accuracy


def main():

    parser = argparse.ArgumentParser(description='softmax regression')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')
    parser.add_argument('--lamb', type=float, default=1, help='Regularization parameter')
    parser.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=config.TRAIN, help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=config.TEST, help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=config.TEST, help='Test data file.')


    args = parser.parse_args()


    x = get_vector(config.TRAIN)
    y = get_label(config.TRAIN)

    datasize = x.shape[0]
    split = int(datasize*0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    W = train_model(x_train, y_train, args)
    acc = test_accuracy(W, x_test, to_classlabel(y_test))
    print(acc)

if __name__ == "__main__":
    main()

