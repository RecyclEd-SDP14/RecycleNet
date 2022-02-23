import sys
import requests


def classify(filename):
    with open(filename, 'rb') as f:
        r = requests.post('http://34.79.127.126:5000/classify', files={'file': f})
        print(r.content.decode('UTF-8'))


if __name__ == '__main__':
    classify(sys.argv[1])
