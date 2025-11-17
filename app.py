from flask import Flask, render_template, request, redirect, url_for, flash
import os, io, base64, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'dev-secret'

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
model = pickle.load(open(os.path.join(MODEL_DIR,'best_model.pkl'),'rb'))
le_target = pickle.load(open(os.path.join(MODEL_DIR,'label_encoder_target.pkl'),'rb'))
le_cuisine = pickle.load(open(os.path.join(MODEL_DIR,'label_encoder_cuisine.pkl'),'rb'))
le_spice = pickle.load(open(os.path.join(MODEL_DIR,'label_encoder_spice.pkl'),'rb'))

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return out

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        if 'csv_file' in request.files and request.files['csv_file'].filename:
            df = pd.read_csv(request.files['csv_file'])
            cols = {
                'cuisine':'Cuisine_top1',
                'spice':'Spice Tolerance',
                'sweet':'Sweet tooth level (1 is low and 5 is high)'
            }
            data = df[[cols['cuisine'],cols['spice'],cols['sweet']]].copy()
            data[cols['cuisine']] = data[cols['cuisine']].astype(str).str.strip()
            data[cols['spice']] = data[cols['spice']].astype(str).str.strip()
            data[cols['sweet']] = pd.to_numeric(data[cols['sweet']], errors='coerce')
            data = data.dropna()

            X = np.vstack([
                le_cuisine.transform(data[cols['cuisine']]),
                le_spice.transform(data[cols['spice']]),
                data[cols['sweet']].astype(float)
            ]).T

            preds = model.predict(X)
            labels = le_target.inverse_transform(preds)
            data['predicted_diet'] = labels

            fig = plt.figure()
            data['predicted_diet'].value_counts().plot(kind='bar')
            plt.title('Predicted Diet Distribution')
            img = fig_to_base64(fig)

            return render_template('results.html',
                                   image_data=img,
                                   table_html=data.head(50).to_html(index=False))

        cuisine = request.form.get('cuisine')
        spice = request.form.get('spice')
        sweet = request.form.get('sweet')
        if cuisine and spice and sweet:
            X = np.array([[
                le_cuisine.transform([cuisine])[0],
                le_spice.transform([spice])[0],
                float(sweet)
            ]])
            pred = model.predict(X)[0]
            label = le_target.inverse_transform([pred])[0]
            return render_template('results.html', single_prediction=label)

    return render_template('index.html',
                           cuisine_options=list(le_cuisine.classes_),
                           spice_options=list(le_spice.classes_))

if __name__=='__main__':
    app.run(debug=True)
