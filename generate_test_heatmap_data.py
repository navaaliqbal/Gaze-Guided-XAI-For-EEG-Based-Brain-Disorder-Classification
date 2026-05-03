import json
import numpy as np

w = 1920
h = 1080
n = 100

x = np.linspace(0, 1, n) * 0.5 + 0.4 
y = np.random.rand(n) * 0.1 + 0.4

d = { 'gaze_data': [] }
for i in range(n):
    d['gaze_data'].append({ 'x': int(x[i] * w), 'y': int(y[i] * h) })

with open('recordings/synthetic.json', 'w') as f:
    json.dump(d, f)