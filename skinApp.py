from flask import Flask,render_template,request,jsonify,url_for
from werkzeug.utils import secure_filename
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image
import numpy as np

from efficientnet_pytorch import EfficientNet


app = Flask(__name__, static_url_path='/static')

### MODEL Densenet201 ###
# Load a pre-trained DenseNet model
# model = models.densenet201(pretrained=False)

# Fine-tune the last few layers of the pre-trained model
# for param in model.features[-1].parameters():
#     param.requires_grad = True
# for param in model.features[-2].parameters():
#     param.requires_grad = True
# for param in model.features[-3].parameters():
#     param.requires_grad = True

# Modify the last fully connected layer to match the number of classes
# num_classes = 14
# in_features = model.classifier.in_features
# model.classifier = nn.Linear(in_features, num_classes)
    
# NUM_CLASSES = 14
# face_classes= ["Dermatitis perioral", "Fungal Acne", "milia", "rosacea", "folikulitis", "Eksim", "herpes", "blackhead", "panu", "kutil filiform", "acne nodules", "flek hitam", "Pustula", "whitehead"]

### End Model DenseNett201 ###

## Model efficience Net
model = EfficientNet.from_pretrained('efficientnet-b7')

# Fine-tune the last few layers of the pre-trained model
for param in model._fc.parameters():
    param.requires_grad = True
    
# Modify the last fully connected layer to match the number of classes
num_classes = 5
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)

face_classes = ['pustula', 'flek hitam', 'rosacea', 'blackhead', 'whitehead']
## End Model

hasil_prediksi  = '(none)'
gambar_prediksi = '(none)'

app.config['UPLOAD_PATH'] = '/static/images/results/'

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

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
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
        test_image = Image.open(f'.{gambar_prediksi}')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(test_image).unsqueeze(0)

        # Make predictions using your loaded model
        with torch.no_grad():
            output = model(input_tensor)

        # Process the predictions
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        predicted_class = predicted_class.item()


        # Prediksi Gambar

        hasil_prediksi = face_classes[int(predicted_class)]

    # Return hasil prediksi dengan format JSON
    return jsonify({
        "prediksi": hasil_prediksi,
        "gambar_prediksi" : gambar_prediksi
    })

# predict product recommendation
def recommendation():
    pass
		

if __name__ == '__main__':
    # Load model yang telah ditraining
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model_skin90.pth'))
    model.to(device)

	# Run Flask di localhost 
    app.run(host='127.0.0.1', port=5001, debug=True)