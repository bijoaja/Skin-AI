from flask import Flask,render_template,request,jsonify

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("index.html")

# =[Main]========================================		

if __name__ == '__main__':

	# Run Flask di localhost 
	app.run(host='127.0.0.1', port=5001, debug=True)