from flask import Flask,request,jsonify
from model import LinearRegression
import numpy as np
import pickle

app = Flask(__name__)

def SalPred(x):
    model = pickle.load(open("checkpoint.pkl","rb"))
    return model.predict(np.array([x]))

@app.route('/api/predict',methods=['POST'])
def predict():
    exp = request.json['experience']
    pred = SalPred(exp)
    return jsonify({'salary' : pred})


if __name__ == "__main__":
    app.run(debug=True)