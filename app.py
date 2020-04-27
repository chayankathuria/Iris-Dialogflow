import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin

app = Flask(__name__)
model = pickle.load(open('rf.pkl', 'rb'))

@app.route('/')
def hello():
    return 'Hello World'

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):

    #sessionID=req.get('responseId')
    result = req.get("queryResult")
    #user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    Petal_length=parameters.get("number")
    Petal_width = parameters.get("number1")
    Sepal_length=parameters.get("number2")
    Sepal_width=parameters.get("number3")
    int_features = [Petal_length,Petal_width,Sepal_length,Sepal_width]
    
    final_features = [np.array(int_features)]
	 
    intent = result.get("intent").get('displayName')
    
    if (intent=='IrisData'):
        prediction = model.predict(final_features)
    
        output = round(prediction[0], 2)
    
    	
        if(output==0):
            flowr = 'Setosa'
    
        if(output==1):
            flowr = 'Versicolour'
        
        if(output==2):
            flowr = 'Virginica'
       
        fulfillmentText= "The Iris type seems to be..  {} !".format(flowr)
        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    #else:
    #    log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)

if __name__ == '__main__':
    app.run()
#if __name__ == '__main__':
#    port = int(os.getenv('PORT', 5000))
#    print("Starting app on port %d" % port)
#    app.run(debug=False, port=port, host='0.0.0.0')
