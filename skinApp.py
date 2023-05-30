from flask import Flask,render_template,request,jsonify
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from modelResnet18 import *
from classNames import className
import pandas as pd
from PIL import Image
import numpy as np
import shutil
from urllib.parse import urlparse

app = Flask(__name__, static_url_path='/static')

##### Model Resnet18 ####
path = "skin-90/"
face_classes = className(path)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 14
model = ResNet(num_classes)
##### End Model Resnet18 ####

## Default Data ##
faceClasses  = 'Normal / Not Detect'
diagnosis = '(none)'
akurasi = '(none)'
gambar_prediksi = '(none)'
data_recomm = '(none)'
skinntone = '(none)'
## End Default Data ##

app.config['UPLOAD_PATH'] = '/static/images/results/'

# Home page
@app.route("/")
def home():
    member= [
        {
            "name": "Clara Adriana",
            "position":"Co-Detection",
            "Image": "Clara Adriana",
            "github":"https://github.com/claraa24",
            "linkedin":"https://www.linkedin.com/in/claraadrianasidauruk"
        },
        {
            "name": "Joel Binsar",
            "position":"Team-Leader",
            "Image": "Joel Binsar",
            "github":"https://github.com/bijoaja",
            "linkedin":"https://www.linkedin.com/in/joelbinsar"
        },
        {
            "name": "Pujaningsih",
            "position":"Co-Detection",
            "Image": "Pujaningsih",
            "github":"https://github.com/Pujaningsih39",
            "linkedin":"https://www.linkedin.com/in/puja-ningsih-088781268"
        },
        {
            "name": "Putri Wulandari",
            "position":"Co-Recommendations",
            "Image": "Putri",
            "github":"https://github.com/putriwulan05",
            "linkedin":"https://www.linkedin.com/in/putri-wulandari-148a1b23b"
        },
        {
            "name": "Wisnu Wijaya",
            "position":"Front-End",
            "Image": "Wisnu",
            "github":"https://github.com/Wisnuoke34",
            "linkedin":"https://www.linkedin.com/in/wisnuwiz"
        },
        {
            "name": "Rita Dwi Pangesti",
            "position":"Co-Recommendations",
            "Image": "Rita",
            "github":"https://github.com/ritapangesti",
            "linkedin":"https://www.linkedin.com/in/ritadwipangesti"
        },
    ]
    default_recom = medication("Normal / Not Detect")
    return render_template("index.html", memberData=member, default_recom=default_recom)

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
    img.convert("RGB")

    
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
		
# For Read Data Medication
def medication(skin_condition):
    df_pd = pd.read_excel("Rekomendasi Obat Penyakit pada Wajah.xlsx", engine="openpyxl")
    data_recomm = df_pd[df_pd["Skin Condition"].apply(lambda x: x.lower()) == skin_condition.lower()]
    data_recomm["Medication"] = data_recomm["Medication"].apply(lambda x: x.split('\n'))
    data_recomm["Skincare Ingredients"] = data_recomm["Skincare Ingredients"].apply(lambda x: x.split('\n'))
    data_recomm["Resources"] = data_recomm["Resources"].apply(lambda x: x.split('\n'))
    data_recomm["Domain"] = data_recomm["Resources"].apply(lambda x: [urlparse(url).netloc for url in x])
    return data_recomm.to_dict(orient='records')

# [Routing untuk API Face Analysis]	
@app.route("/api/faceDetect",methods=['POST'])
def apiDeteksi():

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)
    # Set/mendapatkan extension dan path dari file yg diupload
    gambar_prediksi = f'./static/images/results/{filename}'

    if filename != '':

        # Simpan Gambar
        save_path = os.path.join("static/images/results/", filename)
        uploaded_file.save(save_path)
        
        # Conversi file ke jpg
        filename, extension = os.path.splitext(save_path)
        new_path = f"{filename}.jpg"
        shutil.move(gambar_prediksi, new_path)

        # Predict Image
        probs, classes = predict2(new_path, model)
        probsMax = max(probs)

        if probsMax>0.85:
            faceClasses = face_classes[classes[probs.index(probsMax)]]
        else:
            faceClasses = 'Normal / Not Detect'
        diagnosis = face_classes[classes[probs.index(probsMax)]]
        akurasi = "{:.2f}%".format(probsMax*100)
    else:
        faceClasses = "Upload jpeg file"
    data_recomm = medication(faceClasses)

    # Return hasil prediksi dengan format JSON
    return jsonify({
        "prediksi": faceClasses,
        "diagnosis": diagnosis,
        "akurasi": akurasi,
        "gambar_prediksi" : new_path,
        "data_rekomendasi": data_recomm
    })


if __name__ == '__main__':
    # Load model yang telah ditraining
    try:
        model.load_state_dict(torch.load('best_model2.pth'))
    except RuntimeError as error:
        if 'Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False' in str(error):
            model.load_state_dict(torch.load('best_model2.pth', map_location=torch.device('cpu')))
        else:
            raise error

    

	# Run Flask di localhost 
    app.run(host='127.0.0.1', port=5001, debug=True)