import sys
import requests

def classify(filename):
    with open(filename, 'rb') as f:
        r = requests.post('http://34.79.127.126:5000/foo', files={'file':f})
        print(r.content)

if __name__ == '__main__':
    classify(sys.argv[1])
