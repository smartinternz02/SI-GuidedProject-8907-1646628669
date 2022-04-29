import numpy as np
import pickle
import pandas
import os
from flask import Flask, request,jsonify,render_template


app = Flask(__name__)
model = pickle.load(open(r'Visarf12.pkl', 'rb'))


@app.route('/')# route to display the home page
def home():
    return render_template('Visa_Approval.html') #rendering the home page

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    names = [['FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR','SOC_N']]
    data = pandas.DataFrame(features_values,columns=names)
    

     # predictions using the loaded model file
    prediction=model.predict(data)
    print(prediction)
   
   
    return render_template("resultVA.html",prediction_text =prediction)
    
     # showing the prediction results in a UI
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
