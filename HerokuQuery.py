import requests
import json

resp = json.loads((requests.post("https://recycl-ed.herokuapp.com/predict", files={'file': open('plastic_bottle.jpg', 'rb')})).text)

print()
print("Prediction:", resp["prediction"])
print("Confidence:", resp["confidence"])
print()