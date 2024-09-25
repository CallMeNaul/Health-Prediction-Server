from flask import Flask, render_template, request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import pandas as pd
from keras import models
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from keras.models import load_model

model = models.load_model("my_model_xray.h5")
model1 = models.load_model("my_model.h5")
model2 = load_model("my_model_xray_CV.h5")

app = Flask(__name__)

# ==================Lay du lieu tu firebase==================================
cred = credentials.Certificate("appdacn-1b69f-firebase-adminsdk-1z5tx-3ff1703965.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
list_account=[]
list_id=[]

benhtim=[0,0]
benhtieuduong=[0,0]
benhviemphoi=[0,0]
docs_ref11 = db.collection("Prediction").stream()
for doc in docs_ref11:
    dt=doc._data
    rate=dt.get('rate')
    # print(type(rate))
    type=dt.get('type')
    if (isinstance(rate, list)):
        for i in range(len(rate)):
            if (type[i]=='2'):
                benhviemphoi[1]=1+benhviemphoi[1]
                if (rate[i]>'50'): benhviemphoi[0]=1+benhviemphoi[0]
            elif (type[i]=='1'):
                benhtieuduong[1]=1+benhtieuduong[1]
                if (rate[i]>'50'): benhtieuduong[0]=1+benhtieuduong[0]
            else:
                benhtim[1]=benhtim[1]+1
                if (rate[i]>'50'): benhtim[0]=benhtim[0]+1
print(benhviemphoi)
print(benhtim)
print(benhtieuduong)
docs_ref = db.collection("users").stream()
for doc in docs_ref:
    list_account.append(doc._data.get('gmail'))
    list_id.append(doc.id)
# ======================================================
@app.route('/', methods=['GET'])
def index1():
    return render_template('index.html', my_array=list_account, size=len(list_account),
                           benhtim=benhtim,benhtd=benhtieuduong,benhvp=benhviemphoi)
@app.route('/send_user', methods=['POST'])
def process_item1():
    data = request.get_json()
    selected_item = data.get('email')
    # //------------------lay dâta----------------------------
    index = 0
    for i in range(0, len(list_account)):
        if (list_account[i] == selected_item):
            index = i
            break
    docs_ref1 = db.collection("users").document(list_id[index]).get()
    chuoi = []
    dob=""
    name=""
    job="Bác sĩ"
    avatar_url=""
    addr=""
    sex="nam"
    if docs_ref1.exists:
        data = docs_ref1.to_dict()
        if (str(data.get('sex'))=='1'): sex='nu'
        dob=data.get('DoB')
        addr=data.get('address')
        name=data.get('name')
        avatar_url=data.get('avatarUrl')
        if(str(data.get('isDoctor'))==""):job="Chưa cập nhập"
        list=data.get('Calender')
        # print(name,dob,addr)
        for item in list:
            arr=item.split(';')
            chuoi.append('Có hẹn với '+ arr[2]+' lúc '+arr[1]+' giờ, ngày '+ arr[0])

    # print(jsonify({'sex':sex,'dob':dob,'addr':addr,'name':name,
    #                 'job':job,'avatar_url':avatar_url,'list':chuoi}))
    # -----------------------------------------------------------------
    return jsonify({'sex':sex,'dob':dob,'addr':addr,'name':name,
                    'job':job,'avatar_url':avatar_url,'list':chuoi})
@app.route('/send_email', methods=['POST'])
def process_item():
    data = request.get_json()
    selected_item = data.get('email')
    rates,times,types=Querydatafrom_DB(selected_item)
    d=[]
    for item in types[0]:
        if(item=="0"):
            d.append("bệnh tim")
        elif(item=="1"):
            d.append("bệnh tiểu đường")
        else:
            d.append("bệnh viêm phổi")
    types=[]
    types.append(d)

    return jsonify({'time':times[0],'rate':rates[0],'type':types[0]})
# =======================Lâý dữ liệu từ DB=======================
def Querydatafrom_DB(selected_item):
    index=0
    for i in range(0,len(list_account)):
        if (list_account[i]==selected_item):
            index=i
            break
    try:
        docs_ref1 = db.collection("Prediction").document(list_id[index]).get()
        a = []
        b = []
        c = []
        if docs_ref1.exists:
            data = docs_ref1.to_dict()
            a.append(data.get('rate'))
            b.append(data.get('time'))
            c.append(data.get('type'))
        return a, b, c
    except :
        return [[]],[[]],[[]]
# ===================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if len(data['features']) < 12:
            return jsonify({'error': 'Invalid input format'}), 400
        t = [int(value) for value in data['features']]
        print(t)
        type= [int(value) for value in data['type']]
# ----------------------------------------------------------------------------------------
        if(type[0]==2):
            raw_data = pd.read_csv(r"diabetes_data.csv")
            df = pd.DataFrame([{
                'Age': t[0], 'Sex': t[1], 'HighChol': t[2], 'BMI': t[3],
                'HeartDiseaseorAttack': t[4], 'PhysActivity': t[5],
                'Fruits': t[6], 'Veggies': t[7], 'GenHlth': t[8] + 1, 'PhysHlth': t[9], 'DiffWalk': t[10], 'HighBP': t[11],
                'Diabetes': 0
            }])

            raw_data = raw_data.drop(['CholCheck', 'Smoker', 'HvyAlcoholConsump', 'MentHlth', 'Stroke'], axis=1)
            raw_data = pd.concat([raw_data, df], ignore_index=True)

            columns_to_get_dummies = ['Sex', 'HighChol', 'HighBP', 'PhysActivity', 'Fruits', 'Veggies',
                                      'HeartDiseaseorAttack', 'DiffWalk']
            data = pd.get_dummies(raw_data, columns=columns_to_get_dummies)
            standardScaler = StandardScaler()
            columns_to_scale = ['Age', 'BMI', 'GenHlth', 'PhysHlth']

            data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
            data = data.dropna()
            # print(data.info())
            # print(data.columns)
            bool_columns = ['Age', 'BMI', 'GenHlth', 'PhysHlth', 'Diabetes', 'Sex_0', 'Sex_1',
                            'HighChol_0', 'HighChol_1', 'HighBP_0', 'HighBP_1', 'PhysActivity_0',
                            'PhysActivity_1', 'Fruits_0', 'Fruits_1', 'Veggies_0', 'Veggies_1',
                            'HeartDiseaseorAttack_0', 'HeartDiseaseorAttack_1', 'DiffWalk_0',
                            'DiffWalk_1']
            data[bool_columns] = data[bool_columns].astype(int)
            X_test = data.drop('Diabetes', axis=1)
            X_test = X_test.iloc[[-1]]
            prediction = model.predict(X_test)
            print(int(float(prediction[0, 0]) * 10000))
            return jsonify({'prediction': int(float(prediction[0, 0]) * 10000)})
        else:
            raw_data = pd.read_csv(r"data_heart.csv")
            df = pd.DataFrame([{
                'age': t[0],
                'sex': t[1],
                'cp': t[2],
                'trestbps': t[3],
                'chol': t[4],
                'fbs': t[5],
                'restecg': t[6],
                'thalach': t[7],
                'exang': t[8],
                'oldpeak': t[9] * 1.0,
                'slope': t[10],
                'ca': t[11],
                'thal': t[12],
                'target': 0
            }])
            raw_data = pd.concat([raw_data, df], ignore_index=True)

            columns_to_get_dummies = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            data = pd.get_dummies(raw_data, columns=columns_to_get_dummies)
            standardScaler = StandardScaler()
            columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
            # print(data.info())
            bool_columns = ['sex_0.0', 'sex_1.0', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
                            'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'exang_0',
                            'exang_1', 'slope_0', 'slope_1', 'slope_2', 'ca_0', 'ca_1', 'ca_2', 'ca_3',
                            'ca_4', 'thal_0', 'thal_1', 'thal_2', 'thal_3']

            data[bool_columns] = data[bool_columns].astype(int)
            X_test = data.drop('target', axis=1)
            X_test = X_test.iloc[[-1]]
            prediction = model1.predict(X_test)
            print(int(float(prediction[0, 0]) * 10000))
            return jsonify({'prediction': int(float(prediction[0, 0]) * 10000)})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500
def preprocess_image(image):
    image = tf.image.resize(image, (120, 120))
    image = np.array(image)
    image = image / 255.0
    return image

@app.route('/predict_xray', methods=['POST'])
def predict1():
    image_file = request.files['image']
    image = tf.image.decode_image(image_file.read(), channels=3)
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    prediction = model2.predict(preprocessed_image)
    print(prediction[0][0])
    response = {'prediction': int(prediction[0][0]*10000)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
