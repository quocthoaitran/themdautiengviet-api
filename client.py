import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from predict import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def welcome():
    return jsonify({ 'result': "Welcome to Them dau cho tieng Viet khong dau!!!"})

@app.route("/predict", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    raw_data = req_data['data']
    print(raw_data)
    result = predict(raw_data)
    return jsonify({ 'result': result})

if __name__ == '__main__':
    app.run(host='localhost',port=4000)