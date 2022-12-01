import flask
from flask import render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('Avito_ResultF.xlsx', index_col = None)
data = data.drop(['Unnamed: 0', 'Price'], axis = 1)
app = flask.Flask(__name__, template_folder = 'templates')

def inp_transform(val_dict):
    data.loc[len(data.index)] = list(val_dict.values())
    # preprocess
    cat_cols3 = data[['N_rooms', 'Repairs', 'House_type', 'District']]
    num_cols3 = data.drop(['N_rooms', 'Repairs', 'House_type', 'District'], axis=1)
    std_scaler = StandardScaler()
    scaled = std_scaler.fit_transform(num_cols3)
    scaled_data = pd.DataFrame(data=scaled, columns=num_cols3.columns)
    data_f = scaled_data
    data_f[['N_rooms', 'Repairs', 'House_type', 'District']] = cat_cols3
    data_f = pd.get_dummies(data=data_f, columns=['N_rooms', 'Repairs', 'House_type', 'District'], drop_first=True)
    s = pd.DataFrame(dict(data_f.iloc[-1]), columns = data_f.columns, index = [0])
    return s

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods = ['POST'])

def predict():
    with open('best_regressor.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        s_row = inp_transform(flask.request.form.to_dict())
        y_pred = loaded_model.predict(s_row)
        y_pred = int(y_pred)
        return render_template('main.html', y_pred = y_pred)

if __name__ == "__main__":
    app.run()

