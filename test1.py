import joblib
new_sjsl = joblib.load('/home/dyn/dbz/python sklearn model/sjsl01.pkl')
predict1=new_sjsl.predict([[0,0,24]])
predict1[0]

import joblib
import pandas as pd
new_SARIMAX = joblib.load('/home/dyn/dbz/python sklearn model/SARIMAX01.pkl')

from flask import Flask, redirect, url_for, request

app = Flask(__name__)

@app.route('/api/')
def hello_name():
    id1=float(request.args.get('id1'))
    id2=float(request.args.get('id2'))
    id3=float(request.args.get('id3'))
    return  str(new_sjsl.predict([[id1,id2,id3]])[0])


@app.route('/api/SARIMAX/')
def hello_name2():
    id1=request.args.get('id1')
    id2=request.args.get('id2')
    time1=pd.to_datetime(id1)
    time2=pd.to_datetime(id2)
    predict2=new_SARIMAX.predict(time1,time2)
    json_index=predict2.to_json(orient="index")
    return  json_index


print('1000')

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=8001)
