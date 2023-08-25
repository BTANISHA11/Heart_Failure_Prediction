import uvicorn
from keras.models import model_from_json
from fastapi import FastAPI
from model import Details
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    return {'message': "This is Heart Failure Prediction API."}

@app.post('/prediction')
def api1(req:Details ):
    data = req.dict()
    age = data['age']
    anaemia = data['anaemia']
    creatinine_phosphokinase= data['creatinine_phosphokinase']
    diabetes= data['diabetes']
    ejection_fraction= data['ejection_fraction']
    high_blood_pressure= data['high_blood_pressure']
    platelets= data['platelets']
    serum_creatinine= data['serum_creatinine']
    serum_sodium= data['serum_sodium']
    sex= data['sex']
    smoking= data['smoking']
    time= data['time']
    arr = [age,anaemia,creatinine_phosphokinase,diabetes,
                    ejection_fraction,high_blood_pressure,platelets,
                    serum_creatinine,serum_sodium,sex,smoking,time]
    scaler = load('std_scaler.bin')
    scaler.transform([arr])
    result  = model.predict([arr])

    return {
        "Success":True,
        "Prediction":float(result[0][0])
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000)