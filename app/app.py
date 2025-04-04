import pandas as pd
import pickle
from src.config import *
import csv, os, time, tempfile
from datetime import datetime
from io import TextIOWrapper
from flask import Flask, render_template, request, session, redirect, url_for, flash, Response
import secrets
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder='templates')
app.secret_key = 'LOL'

@app.route('/', methods=['GET', 'POST'])
def render_welcome():
    if request.method == 'GET':
        model_options = {
            "Model 1: CatBoost Classifier": "best_cat_pipe",
            "Model 2: CatBoost Classifier with Calibration": "best_cat_calibrated",
            "Model 3: CatBoost Classifier With Calibration and Threshold 0.6": "prob_best_cat_calb",
        }
        return render_template('welcome.html', model_options= model_options)
    elif request.method == 'POST':
        session['sel_model'] = request.form.get('model')
        return redirect(url_for('render_prediction'))


@app.route('/prediction', methods=['GET', 'POST'])
def render_prediction():
    if not session.get('sel_model'):
        return redirect(url_for('render_welcome'))  # Redirect to welcome page if no model selected
    if request.method == 'GET':
        return render_template('Prediction.html')
    elif request.method == 'POST':
        if request.form.get('input_type') == 'manual':
            print('in POST of prediction -->input type = manual')
            return render_template('Prediction.html', show_manual_form=True)

        elif request.form.get('input_type') == 'csv':
            return render_template('Prediction.html', show_csv_form=True)
        else:
            return "Invalid form submission", 400

@app.route('/manual', methods=['GET', 'POST'])
def manual_form():
    print('in manual_form()')
    if request.method == 'POST':
        sel_model = session.get('sel_model')
        if not sel_model:
            return "No model selected. Please go back and choose a model.", 400
        try:
            if sel_model == 'best_cat_pipe':
                model_path = f'{project_dir}/models/trained_models/best_cat_pipe.pkl'
            elif sel_model in ('best_cat_calibrated', 'prob_best_cat_calb'):
                model_path = f'{project_dir}/models/trained_models/best_cat_calibrated.pkl'
            else:
                return "Invalid model selection", 400
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            return "Model file not found", 500
        except Exception as e:
            return f"Error loading model: {str(e)}", 500

        age = request.form.get('age')
        job = request.form.get('job')
        marital = request.form.get('marital')
        education = request.form.get('education')
        default = request.form.get('default')
        balance = request.form.get('balance')
        housing = request.form.get('housing')
        loan = request.form.get('loan')
        contact = request.form.get('contact')
        day = request.form.get('day')
        month = request.form.get('month')
        duration = request.form.get('duration')
        campaign = request.form.get('campaign')
        pdays = request.form.get('pdays')
        previous = request.form.get('previous')
        poutcome = request.form.get('poutcome')

        input_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }
        print(input_data)
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            if prediction == 1: msg = "✅ Client WILL subscribe to a term deposit"
            else: msg = "❌ Client will NOT subscribe to a term deposit"

            return render_template('Prediction.html', show_manual_prediction=True, prediction=msg)
                # return "✅ Client WILL subscribe to a term deposit"


        except Exception as e:
            return f"Prediction error: {str(e)}", 500

    elif request.method == 'GET':
        return render_template('Prediction.html', show_manual_form=True)

@app.route('/csv', methods=['POST'])
def csv_form():
    if request.method == 'POST':
        sel_model = session.get('sel_model')
        uploaded_csv = request.files['csvfile']
        try:
            # 1. Header check (no headers)
            sample = uploaded_csv.read(1024).decode('utf-8')
            uploaded_csv.stream.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            if csv.Sniffer().has_header(sample): print("CSV header confirmed, sniffer check PASS...!")
            else:
                flash('Error: 400 Bad Request: Invalid CSV headers - sniffer check failed.')
                return redirect(url_for('render_prediction'))

            # 2. Read directly into pandas
            df = pd.read_csv(TextIOWrapper(uploaded_csv.stream), delimiter=dialect.delimiter)
            if not df.empty: print('Pandas check PASS...!')
            else:
                flash('Error 400 Bad Request: Invalid Dataframe - Pandas df check failed.')
                return redirect(url_for('render_prediction'))

            # 3. Quick column check (if needed)
            if all(col in df.columns for col in total_features):
                print('Columns check PASS...!')
            else:
                flash('Error: 400 Bad Request: Invalid features in CSV - Columns check failed.')
                return redirect(url_for('render_prediction'))

        except pd.errors.EmptyDataError:
            flash('Error: 400 Empty file')
            return redirect(url_for('render_prediction'))
        except pd.errors.ParserError:
            flash('Error: 400 Invalid CSV format')
            return redirect(url_for('render_prediction'))
        except Exception as e:
            flash(f'Error: 400 {str(e)}')
            return redirect(url_for('render_prediction'))
        except uploaded_csv is None:
            flash('Error: 400 Empty input')
            return redirect(url_for('render_prediction'))

        file_id = secrets.token_hex(8)
        session['prediction_file_id'] = file_id  # Only stores a small ID

        if sel_model == 'best_cat_pipe':
            with open(f'{project_dir}/models/trained_models/best_cat_pipe.pkl', 'rb') as f:
                model = pickle.load(f)
                prediction = model.predict(df)
                df['Predictions'] = prediction
                # session['result_csv'] = df.to_csv(index=False)
                df.to_csv(f'temp/temp_{file_id}.csv', index=False)
                return render_template('Prediction.html', show_download=True)
        elif sel_model == 'best_cat_calibrated' or sel_model == 'prob_best_cat_calb':
            with open(f'{project_dir}/models/trained_models/best_cat_calibrated.pkl', 'rb') as f:
                model = pickle.load(f)
                prediction_prob = model.predict_proba(df)[:, 1]
                prediction = (prediction_prob >= 0.6).astype(int)
                df['Predictions'] = prediction
                # session['result_csv'] = df.to_csv(index=False)
                df.to_csv(f'temp/temp_{file_id}.csv', index=False)
                return render_template('Prediction.html', show_download=True)

@app.route('/download', methods=['POST'])
def download_file():
    file_id = session.get('prediction_file_id')
    result_csv = pd.read_csv(f'temp/temp_{file_id}.csv')

    # Cleanup old files
    for file in os.listdir('temp'):
        if file.startswith('temp_') and file.endswith('.csv'):
            filepath = os.path.join('temp', file)
            if (time.time() - os.path.getmtime(filepath)) > 300:
                try:
                    os.remove(filepath)
                except:
                    pass

    if not result_csv.empty:
        filename = f'Predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return Response(
            result_csv.to_csv(index=False),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    else:
        return "No predictions found", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)