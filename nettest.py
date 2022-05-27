import os
import requests
import time
import numpy as np

tgt = os.path.join(os.getcwd(), 'data', 'trashnet-filtered', 'trash')

def go():
    foo = []
    for fname in os.listdir(tgt):
        f = os.path.join(tgt, fname)
        extension = os.path.splitext(f)[1]
        if os.path.isfile(f) and extension == '.jpg':
            with open(f, 'rb') as fi:
                t1 = time.time()
                r = requests.post('http://34.79.127.126:5000/classify', files={'file': fi})
                print(r.content.decode('UTF-8'))
                t2 = time.time()
                td = t2 - t1
                foo.append(td)
    print("Number of requests: " + str(len(foo)))
    print("Average: " + str(np.mean(foo)) + "s")
    print("Std: " + str(np.std(foo)) + "s")
    print("Max: " + str(np.max(foo)) + "s")
    print("Min: " + str(np.min(foo)) + "s")

if __name__ == '__main__':
    go()
