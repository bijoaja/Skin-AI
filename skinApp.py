from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np

app = Flask(__name__, static_url_path='/static')

##### Model Resnet18 ####

face_classes = ['Dermatitis perioral', 'Eksim', 'Pustula', 'acne nodules', 'blackhead', 'flek hitam', 'folikulitis', 'fungal acne', 'herpes', 'kutil filiform', 'milia', 'panu', 'rosacea', 'whitehead']

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(face_classes))
model = model_ft.to(device)

##### End Model Resnet18 ####

hasil_prediksi  = '(none)'
gambar_prediksi = '(none)'

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
		
# [Routing untuk API]	
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
        hasil_prediksi = classes[probs.index(probsMax)]

    # Return hasil prediksi dengan format JSON
    return jsonify({
        "prediksi": face_classes[int(hasil_prediksi)],
        "gambar_prediksi" : gambar_prediksi
    })

# predict product recommendation
def recommendation():
    pass
		

if __name__ == '__main__':
    # Load model yang telah ditraining
    model.load_state_dict(torch.load('model_resnet18.pth'))
    model.to(device)

	# Run Flask di localhost 
    app.run(host='127.0.0.1', port=5001, debug=True)