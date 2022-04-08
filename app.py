import uvicorn
import gunicorn
from fastapi import FastAPI, encoders
from Bankcredit import BankCredit
import pickle
import pandas as pd
from starlette import responses

# création obj app:
app = FastAPI()

model_pkl = open('data/model.pkl', 'rb')
model = pickle.load(model_pkl)

lime_df = pd.read_csv('data/lime_df_trim.csv', index_col=0)

@app.get('/')
def index():
    return {'Text' : "L'API est lancée"}

@app.post('/predict')
def predict(data: BankCredit):
    """
    Fonction predict qui prend les infos du client sous forme json et retourne
    la décision et la probabilité
    """

    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])

    prediction = pd.DataFrame(model.predict_proba(data_df))

    json_item = encoders.jsonable_encoder(prediction)
    return responses.JSONResponse(content=json_item)


@app.get('/gal_exp/{input_id}')
def gal_exp(input_id : int):

    data_df = lime_df[lime_df.index == input_id]
    data_df.reset_index(inplace=True, drop=True)

    data_df = data_df.ticks.str.split(':', n=1, expand=True)

    json_item = encoders.jsonable_encoder(data_df)
    return responses.JSONResponse(content=json_item)

if __name__ == '__main__':
    uvicorn.run(app)