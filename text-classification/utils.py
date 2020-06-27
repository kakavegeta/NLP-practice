import numpy as np

def onehot_encode(y):
    return (np.arange(np.max(y)+1) == y[:,None]).astype(float)

def softmax(z):
    return (np.exp(z.T)/np.sum(np.exp(z), axis=1)).T

def cross_entropy(y, y_):
    # y: target; y_: output
    return -np.sum(y * np.log(y_), axis=1)

def cost(y, y_):
    return np.mean(cross_entropy(y, y_))


def main():
    y = np.array([0,1,1,2,2,3])
    y = onehot_encode(y)
    print(y)

    z = np.array([[1,3,4,5],[2,4,3,5],[4,3,2,1]])
    z = softmax(z)
    print(z)
                
if __name__ == "__main__":
    main()