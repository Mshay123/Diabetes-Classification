import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import joblib
import tensorflow as tf

from flask import Flask, render_template, request
import xgboost as xgb
import pandas as pd
import numpy as np
import dice_ml
import shap
from google import genai

app = Flask(__name__)

# --- טעינת המודלים והסקיילרים ---
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('diabetes_xgboost.json')

dl_model = tf.keras.models.load_model('model.keras')
unsupervised_model = joblib.load('unsupervised.pkl')

try:
    scaler_bmi = joblib.load('scaler_bmi.pkl')
    scaler_all = joblib.load('scaler_all_features.pkl')
    SCALERS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load scalers: {e}")
    SCALERS_LOADED = False


def get_age_category(age):
    if age < 25:
        return 1
    elif age < 30:
        return 2
    elif age < 35:
        return 3
    elif age < 40:
        return 4
    elif age < 45:
        return 5
    elif age < 50:
        return 6
    elif age < 55:
        return 7
    elif age < 60:
        return 8
    elif age < 65:
        return 9
    elif age < 70:
        return 10
    elif age < 75:
        return 11
    elif age < 80:
        return 12
    else:
        return 13


UNSUP_FEATURES = ['HighBP', 'HighChol', 'Age', 'BMI', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits', 'HvyAlcoholConsump',
                  'Education']

FEATURES = [
    {"name": "HighBP", "label": "לחץ דם גבוה", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם אובחן אצלך בעבר לחץ דם גבוה?"},
    {"name": "HighChol", "label": "כולסטרול גבוה", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם אובחן אצלך כולסטרול גבוה?"},
    {"name": "CholCheck", "label": "בדיקת כולסטרול (5 שנים)", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם ביצעת בדיקת כולסטרול ב-5 השנים האחרונות?"},
    {"name": "Stroke", "label": "היסטוריית שבץ", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם עברת אירוע שבץ מוחי בעבר?"},
    {"name": "HeartDiseaseorAttack", "label": "מחלת לב / התקף לב", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם אובחנה אצלך מחלת לב או שעברת התקף לב בעבר?"},
    {"name": "DiffWalk", "label": "קשיי הליכה/ניידות", "type": "switch", "category": "מצבים רפואיים",
     "tooltip": "האם יש לך קושי משמעותי בהליכה או בניידות?"},

    {"name": "Sex", "label": "מין", "type": "select", "options": [(1, "גבר"), (0, "אישה")], "category": "מדדים אישיים",
     "tooltip": "המין שלך."},
    {"name": "Age", "label": "גיל (בשנים)", "type": "number", "min": 18, "max": 120, "category": "מדדים אישיים",
     "tooltip": "הזן/י את הגיל שלך בשנים. המערכת תמיר זאת אוטומטית לקבוצת הגיל המתאימה למודל."},
    {"name": "BMI", "label": "מדד מסת גוף (BMI)", "type": "number", "min": 10, "max": 99, "category": "מדדים אישיים",
     "tooltip": "מדד מסת הגוף שלך (היחס בין משקל לגובה)."},
    {"name": "GenHlth", "label": "בריאות כללית", "type": "select",
     "options": [(1, "מצוין"), (2, "טוב"), (3, "סביר"), (4, "גרוע"), (5, "מאוד גרוע")], "category": "מדדים אישיים",
     "tooltip": "איך היית מדרג/ת את הבריאות הכללית שלך?"},

    {"name": "Smoker", "label": "מעשן/ת", "type": "switch", "category": "אורח חיים",
     "tooltip": "האם עישנת לפחות 100 סיגריות במהלך חייך?"},
    {"name": "PhysActivity", "label": "פעילות גופנית", "type": "switch", "category": "אורח חיים",
     "tooltip": "האם עסקת בפעילות גופנית סדירה לאחרונה?"},
    {"name": "Fruits", "label": "צריכת פירות קבועה", "type": "switch", "category": "אורח חיים",
     "tooltip": "האם את/ה צורך/ת פירות באופן יומיומי?"},
    {"name": "Veggies", "label": "צריכת ירקות קבועה", "type": "switch", "category": "אורח חיים",
     "tooltip": "האם את/ה צורך/ת ירקות באופן יומיומי?"},
    {"name": "HvyAlcoholConsump", "label": "צריכת אלכוהול כבדה", "type": "switch", "category": "אורח חיים",
     "tooltip": "האם את/ה צורך/ת אלכוהול בכמויות גדולות או לעיתים תכופות?"},

    {"name": "MentHlth", "label": "ימי בריאות נפש ירודה", "type": "number", "min": 0, "max": 30,
     "category": "חברה וגישה לרפואה", "tooltip": "בכמה ימים במהלך החודש האחרון חשת שהבריאות הנפשית שלך ירודה?"},
    {"name": "PhysHlth", "label": "ימי בריאות פיזית ירודה", "type": "number", "min": 0, "max": 30,
     "category": "חברה וגישה לרפואה", "tooltip": "בכמה ימים במהלך החודש האחרון חשת שהבריאות הפיזית שלך ירודה?"},
    {"name": "AnyHealthcare", "label": "ביטוח בריאות", "type": "switch", "category": "חברה וגישה לרפואה",
     "tooltip": "האם יש לך גישה לשירותי בריאות או ביטוח רפואי?"},
    {"name": "NoDocbcCost", "label": "ויתור על רופא עקב עלות", "type": "switch", "category": "חברה וגישה לרפואה",
     "tooltip": "האם נמנעת לאחרונה מביקור אצל רופא בגלל העלות?"},
    {"name": "Education", "label": "רמת השכלה", "type": "select",
     "options": [(1, "לא סיימו תיכון"), (2, "בגרות"), (3, "תואר ראשון"), (4, "תארים מתקדמים")],
     "category": "חברה וגישה לרפואה", "tooltip": "מהי רמת ההשכלה שלך?"},
    {"name": "Income", "label": "רמת הכנסה", "type": "select",
     "options": [(1, "פחות מ-25,000$"), (2, "25,000$-35,000$"), (3, "35,000$-50,000$"), (4, "50,000$-75,000$"),
                 (5, "75,000$+")], "category": "חברה וגישה לרפואה", "tooltip": "מהי רמת ההכנסה השנתית הממוצעת שלך?"}
]

for f in FEATURES:
    if f["name"] in UNSUP_FEATURES:
        f["models"] = ["xgboost", "dl", "unsupervised"]
    else:
        f["models"] = ["xgboost", "dl"]

dummy_data = {feat['name']: np.random.randint(0, 2, 100) if feat['type'] == 'switch' else np.random.randint(1, 5, 100)
              for feat in FEATURES}
dummy_data['BMI'] = np.random.randint(15, 40, 100)
dummy_data['Age'] = np.random.randint(1, 14, 100)
dummy_data['Diabetes_binary'] = np.random.randint(0, 2, 100)
df_dice_train = pd.DataFrame(dummy_data).astype(float)


class XGBoostDiCE:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return self.model.predict(data.astype(float))

    def predict_proba(self, data):
        return self.model.predict_proba(data.astype(float))


continuous_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
d = dice_ml.Data(dataframe=df_dice_train, continuous_features=continuous_features, outcome_name='Diabetes_binary')
m = dice_ml.Model(model=XGBoostDiCE(xgb_model), backend="sklearn")
exp = dice_ml.Dice(d, m, method="genetic")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    cluster_id = None
    cluster_name = None
    selected_model = None
    gemini_report = None
    shap_plot_url = None

    categories = {}
    for f in FEATURES:
        cat = f.get("category", "אחר")
        if cat not in categories: categories[cat] = []
        categories[cat].append(f)

    if request.method == 'POST':
        selected_model = request.form.get('model_choice')

        input_data = {}
        original_feature_order = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
                                  "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                                  "NoDocbcCost",
                                  "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

        for feat_name in original_feature_order:
            val = request.form.get(feat_name)
            numeric_val = float(val) if val is not None and val != '' else 0.0
            if feat_name == "Age" and val is not None and val != '':
                numeric_val = float(get_age_category(int(numeric_val)))
            input_data[feat_name] = [numeric_val]

        df_input = pd.DataFrame(input_data).astype(float)

        if selected_model == 'xgboost':
            pred = xgb_model.predict(df_input)[0]
            prob = xgb_model.predict_proba(df_input)[0][1] * 100
            prediction = int(pred)
            probability = round(prob, 2)

            try:
                explainer = shap.Explainer(xgb_model)
                shap_values = explainer(df_input)
                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                img = io.BytesIO()
                plt.tight_layout()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                shap_plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
            except Exception as e:
                print(f"SHAP Error: {e}")

            if prediction == 1:
                try:
                    dice_exp = exp.generate_counterfactuals(
                        df_input, total_CFs=1, desired_class=0,
                        features_to_vary=['HighBP', 'HighChol', 'BMI', 'PhysActivity', 'Fruits', 'Veggies', 'PhysHlth',
                                          'MentHlth']
                    )
                    new_life = dice_exp.cf_examples_list[0].final_cfs_df
                    client = genai.Client(api_key="AIzaSyA6L2n_HL2fpzyQNKDXDsoxS3FTRco_79A")
                    content = f"""
                    You are an AI medical assistant...
                    Current Health Data: {df_input.to_string()}
                    Target Lifestyle: {new_life.to_string()}
                    Instructions: Write in Hebrew. Sections: מצב נוכחי, שינויים מומלצים, צפי שיפור.
                    End with: "אני סייע רפואי מבוסס בינה מלאכותית ואיני רופא. דוח זה הינו להעשרה ומידע בלבד."
                    """
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=content)
                    gemini_report = response.text
                except Exception as e:
                    gemini_report = f"שגיאה בהפקת הדוח החכם: {str(e)}"

        elif selected_model == 'dl':
            df_dl = df_input.copy()

            if SCALERS_LOADED:
                bmi_df = pd.DataFrame(df_dl['BMI'].values, columns=['BMI'])
                df_dl['BMI_scaled'] = scaler_bmi.transform(bmi_df)
            else:
                df_dl['BMI_scaled'] = df_dl['BMI']

            df_dl['LifeThreateningCondition'] = ((df_dl['HeartDiseaseorAttack'] == 1) | (df_dl['Stroke'] == 1)).astype(
                int)
            df_dl['HealthyLifestyleScore'] = (1 - df_dl['Smoker']) + df_dl['Fruits'] + \
                                             df_dl['Veggies'] + df_dl['PhysActivity'] + \
                                             (1 - df_dl['HvyAlcoholConsump'])
            df_dl['SocioEconomicStatus'] = df_dl['Income'] + df_dl['Education']

            cols_to_drop = ['BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
                            'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Education', 'Income']
            df_dl = df_dl.drop(cols_to_drop, axis=1)

            dl_features_order = ["HighBP", "HighChol", "CholCheck", "AnyHealthcare", "NoDocbcCost",
                                 "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
                                 "BMI_scaled", "LifeThreateningCondition", "HealthyLifestyleScore",
                                 "SocioEconomicStatus"]
            df_dl = df_dl[dl_features_order]

            if SCALERS_LOADED:
                dl_input_array = scaler_all.transform(df_dl)
            else:
                dl_input_array = df_dl.values

            pred_prob = dl_model.predict(dl_input_array)[0][0]
            prediction = 1 if pred_prob > 0.5 else 0
            probability = round(pred_prob * 100, 2)

        elif selected_model == 'unsupervised':
            df_unsup = df_input[UNSUP_FEATURES].copy()

            # נרמול ידני לפי המודל שלך
            df_unsup['BMI'] = (df_unsup['BMI'] - 30.00465028) / 7.22761145
            df_unsup['Age'] = (df_unsup['Age'] - 8.61841393) / 2.85188826

            raw_cluster = unsupervised_model.predict(df_unsup)[0]

            # --- התיקון שביצענו: לוגיקה תקינה למרכזי הכובד שלך ---
            # קלאסטר 0 במודל = צעירים, בריאים -> נהפוך לקלאסטר 0
            # קלאסטר 1 במודל = BMI גבוה, השמנה, גיל ביניים -> נהפוך לקלאסטר 2 (סוג 2)
            # קלאסטר 2 במודל = מבוגרים, BMI נמוך -> נהפוך לקלאסטר 1 (סוג 1)
            swap_map = {0: 0, 1: 2, 2: 1}
            cluster_id = swap_map.get(int(raw_cluster), int(raw_cluster))

            cluster_names = {
                0: "בריא (ללא סוכרת)",
                1: "סוכרת סוג 1",
                2: "סוכרת סוג 2"
            }
            cluster_name = cluster_names.get(cluster_id, f"קלאסטר {cluster_id}")

    return render_template('index.html', categories=categories, prediction=prediction, probability=probability,
                           cluster_name=cluster_name, selected_model=selected_model,
                           gemini_report=gemini_report, shap_plot_url=shap_plot_url)


if __name__ == '__main__':
    app.run(debug=True)