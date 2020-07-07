import os

PATH = os.path.dirname(os.path.abspath(__file__)) # task1/

DATAPATH = os.path.join(PATH, "data")
TRAIN = os.path.join(DATAPATH, "train.tsv")
TEST = os.path.join(DATAPATH, "test.tsv")




if __name__ == "__main__":
    print(PATH)
    print(DATAPATH)
    print(TRAIN)
    print(TEST)