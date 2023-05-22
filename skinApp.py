from flask import Flask,render_template,request,jsonify
import os
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import numpy as np

app = Flask(__name__, static_url_path='/static')

##### Model Resnet18 ####

face_classes = ['Dermatitis perioral', 'Eksim', 'Pustula', 'acne nodules', 'blackhead', 'Flek hitam', 'folikulitis', 'fungal acne', 'herpes', 'kutil filiform', 'milia', 'panu', 'rosacea', 'whitehead']

# blackhead, flek hitam, 

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(face_classes))
model = model_ft.to(device)

##### End Model Resnet18 ####

hasil_prediksi  = '(none)'
gambar_prediksi = '(none)'
data_recomm = []
skinntone = '(none)'

app.config['UPLOAD_PATH'] = '/static/images/results/'

# Home page
@app.route("/")
def home():
    img_url = os.listdir('static/images/members')
    member= [
        {
            "name": "Clara Adriana",
            "position":"Co-Detection",
            "github":"https://github.com/claraa24",
            "linkedin":"https://www.linkedin.com/in/claraadrianasidauruk"
        },
        {
            "name": "Joel Binsar",
            "position":"Team-Leader",
            "github":"https://github.com/bijoaja",
            "linkedin":"https://www.linkedin.com/in/joelbinsar"
        },
        {
            "name": "Pujaningsih",
            "position":"Co-Detection",
            "github":"https://github.com/Pujaningsih39",
            "linkedin":"https://www.linkedin.com/in/puja-ningsih-088781268"
        },
        {
            "name": "Putri Wulandari",
            "position":"Co-Recommendations",
            "github":"https://github.com/putriwulan05",
            "linkedin":"https://www.linkedin.com/in/putri-wulandari-148a1b23b"
        },
        {
            "name": "Wisnu Wijaya",
            "position":"Front-End",
            "github":"https://github.com/Wisnuoke34",
            "linkedin":"https://www.linkedin.com/in/wisnuwiz"
        },
        {
            "name": "Rita Dwi Pangesti",
            "position":"Co-Recommendations",
            "github":"https://github.com/ritapangesti",
            "linkedin":"https://www.linkedin.com/in/ritadwipangesti"
        },
    ]
    return render_template("index.html", imagesMember=img_url, memberData=member)

# Proses image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

# predict product recommendation
def predict2(uploaded_file_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(uploaded_file_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)
		
# [Routing untuk API Face Analysis]	
@app.route("/api/faceDetect",methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)

    if filename != '':
    	
        # Set/mendapatkan extension dan path dari file yg diupload
        gambar_prediksi = f'/static/images/results/{filename}'

        # Simpan Gambar
        save_path = os.path.join("static/images/results/", filename)
        uploaded_file.save(save_path)

        # Memuat Gambar

        probs, classes = predict2(f'.{gambar_prediksi}', model_ft.to(device))
        probsMax = max(probs)
        if probsMax>0.9:
            faceClasses = face_classes[classes[probs.index(probsMax)]]
        else:
            faceClasses = 'Normal / Not Detect'
        
        skinT = selected_value(skinntone)
        recomm = recommendation(faceClasses, skinT)
        
        for index, row in recomm.iterrows():
            # Buat objek baru dan tambahkan nilai dari setiap kolom
            obj = {
                'Product': row['Product'],
                'Product_Url': row['Product_Url']
            }
            # Tambahkan objek ke dalam array_of_objects
            data_recomm.append(obj)
        
    print("From Api Detect: ",obj)
    # Return hasil prediksi dengan format JSON
    return jsonify({
        "prediksi": faceClasses,
        "gambar_prediksi" : gambar_prediksi,
        "data_rekomendasi": data_recomm
    })

@app.route("/skinTone", methods=['POST'])
def skinTone():
    global skinntone
    skinntone = request.form.get('value')
    return skinntone
    
def selected_value(value):
    value = skinntone
    return value

# Recommendation Product
def recommendation(skintype, skintone):  # sourcery skip: avoid-builtin-shadow
    # Load the dataframe
    df = pd.read_csv('data_product_recommendation.csv', index_col=[0])
    
    # Create Condition For Cek skintype & skintone already exist in dataset
    ## PROCESS
    # Filter the dataframe based on the given features
    ddf = df[(df['Skin_Type'] == skintype) & (df['Skin_Tone'] == skintone)]

    # Encode categorical features
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(df[['Skin_Type', 'Skin_Tone']])

    # Prepare training data
    X_train = encoded_features
    y_train = df['Good_Stuff']

    # Fit the model with training data
    modelL.fit(X_train, y_train)

    # Make predictions using the fitted model
    encoded_new_data = encoder.transform(ddf[['Skin_Type', 'Skin_Tone']])
    prediksi = modelL.predict(encoded_new_data)

    # Use the prediction to recommend products
    recommendations = ddf[ddf['Rating_Stars'].notnull()]
    recommendations['Prediction'] = prediksi
    recommendations = recommendations.sort_values(by=['Prediction', 'Rating_Stars', 'Price', 'Good_Stuff'], ascending=[False, False, True, False])

    # Create a new dataframe and reset the index from 1
    result = pd.DataFrame(recommendations[['Product', 'Brand', 'Category', 'Price', 'Good_Stuff', 'Ingredients', 'Rating_Stars', 'Product_Url']]).reset_index(drop=True)
    result.index += 1

    # Remove duplicate data and filter by ingredients
    result = result.drop_duplicates(subset="Product")
    filter = result["Ingredients"] != "No Info"
    result = result[filter]
    
    return result
		

if __name__ == '__main__':
    # Load model yang telah ditraining
    model.load_state_dict(torch.load('model_resnet18.pth'))
    model.to(device)
    
    modelL = pickle.load(open("logisticRegression.pkl", "rb"))


	# Run Flask di localhost 
    app.run(host='127.0.0.1', port=5001, debug=True)