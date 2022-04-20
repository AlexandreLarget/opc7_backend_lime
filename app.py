import uvicorn
import gunicorn
from fastapi import FastAPI, encoders
import pickle
import pandas as pd
from starlette import responses
import lime
import dill

# création obj app:
app = FastAPI()


model_pkl = open('data/model.pkl', 'rb')
model = pickle.load(model_pkl)

explainer_pkl = open('data/explainer_lime_2.pkl', 'rb')
explainer = dill.load(explainer_pkl)

def get_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df

data = get_data("data/data_scaled_sample.csv")


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

    data_client = data[data.SK_ID_CURR == input_id]
    data_client = data_client.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    explaination = explainer.explain_instance(data_client.squeeze(axis=0), model.predict_proba, num_features=10)

    liste_n = explaination.as_list()
    liste_n = liste_n

    features = []
    values = []

    for i, j in liste_n:
        features.append(i)
        values.append(j)
    exp = pd.DataFrame(values, index=features, columns=['valeur'])

    exp['copie'] = exp.index
    exp[['copie', 'temp']] = exp.copie.str.rsplit('<', n=1, expand=True)
    try:
        exp[['copie', 'temp']] = exp['copie'].str.rsplit('>', n=1, expand=True)
    except:
        pass
    exp['ticks'] = exp['copie'] + " : " + round(exp['valeur'], 4).astype(str)
    exp.drop(['temp', 'copie'], axis=1, inplace=True)

    exp_2 = exp.sort_values('valeur', ascending=True, key=abs)[-10:]
    exp_2.reset_index(inplace=True, drop=True)

    json_item = encoders.jsonable_encoder(exp_2)
    return responses.JSONResponse(content=json_item)


@app.get('/stats/{input_id}')
def stats(input_id : int):

    data_client = data[data.SK_ID_CURR == input_id]
    data_client = data_client.drop(['SK_ID_CURR', 'TARGET'], axis=1)

    json_item = encoders.jsonable_encoder(data_client)
    return responses.JSONResponse(content=json_item)


if __name__ == '__main__':
    uvicorn.run(app)