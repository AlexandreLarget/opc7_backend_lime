import uvicorn
import gunicorn
from fastapi import FastAPI, encoders
import pickle
import pandas as pd
from starlette import responses

# création obj app:
app = FastAPI()


model_pkl = open('data/model.pkl', 'rb')
model = pickle.load(model_pkl)

def get_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df

data = get_data("data/data_scaled_sample.csv")

lime_df = get_data("data/lime_df.csv")


@app.get('/')
def index():
    return {'Text' : "L'API est lancée"}

@app.get('/predict/{input_id}')
def predict(input_id : int):

    data_client = data[data.SK_ID_CURR == input_id]
    data_client = data_client.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    prediction = pd.DataFrame(model.predict_proba(data_client))
    prediction.reset_index(inplace=True, drop=True)

    json_item = encoders.jsonable_encoder(prediction)
    return responses.JSONResponse(content=json_item)


@app.get('/graph/{input_id}')
def graph(input_id : int):

    data = lime_df[lime_df.index == input_id]
    data.reset_index(inplace=True, drop=True)

    json_item = encoders.jsonable_encoder(data)
    return responses.JSONResponse(content=json_item)


@app.get('/stats/{input_id}')
def stats(input_id : int):

    data_client = data[data.SK_ID_CURR == input_id]
    data_client = data_client.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    json_item = encoders.jsonable_encoder(data_client)
    return responses.JSONResponse(content=json_item)


if __name__ == '__main__':
    uvicorn.run(app)