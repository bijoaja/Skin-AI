from flask import Flask,render_template,request,jsonify,url_for
import os


app = Flask(__name__, static_url_path='/static')

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

# save image to directory
@app.route("/detected/",  methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(f'static/images/results/{file.filename}')
        return 'File berhasil diunggah dan disimpan.'

# predict new image 
def detected():
    pass

# predict product recommendation
def recommendation():
    pass

# =[Main]========================================		

if __name__ == '__main__':

	# Run Flask di localhost 
	app.run(host='127.0.0.1', port=5001, debug=True)