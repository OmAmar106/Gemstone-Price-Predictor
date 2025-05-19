from flask import Flask, redirect, render_template, request, jsonify
from flask import session
from model.predict import predict

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.sqlite3"
# db.init_app(app)
app.app_context().push()
app.secret_key = "APtlnuRu04uv"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
# with app.app_context():
    # db.create_all()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict1():
    d = request.get_json()
    for i in d:
        try:
            d[i] = int(d[i])
        except:
            pass
    k = predict(d['carat'],d['cut'],d['clarity'],d['table'],d['x'],d['y'],d['z'])
    s = f'{k:.2f}'
    return jsonify({"ans":s}), 200

if __name__ == '__main__':
    app.run(debug=True)