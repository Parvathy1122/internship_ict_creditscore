from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)
#model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        
        #model = pickle.load('model70.pkl','rb')
        # create file object with permissions
        with open('/workspaces/internship_ict_creditscore/model70.pkl', 'rb') as f:
    # load using pickle de-serializer
            model = pickle.load(f)
    
        
        # Get feature inputs from the form
        Annual_Income = float(request.form["Annual_Income"])
        Monthly_Inhand_Salary = float(request.form["Monthly_Inhand_Salary"])
        Num_Bank_Accounts = float(request.form["Num_Bank_Accounts"])
        Num_Credit_Card = float(request.form["Num_Credit_Card"])
        Interest_Rate = float(request.form["Interest_Rate"])
        Num_of_Loan = float(request.form["Num_of_Loan"])
        Delay_from_due_date = float(request.form["Delay_from_due_date"])
        Num_of_Delayed_Payment = float(request.form["Num_of_Delayed_Payment"])
        Credit_Mix = int(request.form["Credit_Mix"])
        Outstanding_Debt = float(request.form["Outstanding_Debt"])
        Credit_History_Age = float(request.form["Credit_History_Age"])
        Monthly_Balance = float(request.form["Monthly_Balance"])
        
        

        # Make a prediction using the trained model
        input_features = [[Annual_Income,Monthly_Inhand_Salary,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Credit_Mix,Outstanding_Debt,Credit_History_Age,Monthly_Balance]]
        print('HI',input_features)

        prediction = model.predict(input_features)
        #print("hello",prediction)

        #result = model.predict(input_features)
        #return render_template('result.html', outcome=f'THE RESULT IS {prediction}')
        if prediction==2:
            return render_template('result.html', outcome=f'Good')                            
        elif prediction==1:
            return render_template('result.html', outcome='Standard')
        elif prediction==0:
            return render_template('result.html', outcome='Poor')
if __name__ == '__main__':
    app.run(port=8080, debug=True)
 