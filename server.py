import argparse
from flask import Flask, flash, request, redirect
from utils import delimiter
import flasknet
import os
import secrets
from werkzeug.utils import secure_filename
from waitress import serve
import smtplib
from email.message import EmailMessage

parser = argparse.ArgumentParser(description='RecycleNet server runner')
parser.add_argument('--debug', action='store_true', help="Use dev server")
parser.add_argument('--new', action='store_true', help="Use new classification")
args = parser.parse_args()
s = smtplib.SMTP('localhost')
if args.new:
    folder = "NewData"
else:
    folder = "OldData"


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), UPLOAD_FOLDER)
app.config['SECRET_KEY'] = secrets.token_hex(256)
net = flasknet.FlaskNet(args.new, True, True, 'save' + delimiter() + folder + delimiter() + 'model_best.pth.tar')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    return os.getcwd()

@app.route('/coupon')
def email():
    uid = request.args.get('uid')
    return '''
    <!doctype html>
    <title>Coupon Email Thing</title>
    <h1>Hello! This page isn't fully implemented yet!</h1><br>
    <h2>Enter your email here to get your voucher.</h2>
    <form method="POST">
    <input name="text">
    <input type="submit">
    </form>
    Your user ID is: 
    ''' + uid

@app.route('/coupon', methods=['POST'])
def email2():
    uid = request.args.get('uid')
    msg = EmailMessage()
    msg.set_content('Thanks for using RecyclED. This coupon is totes worth £0.10. (Transaction id: ' + uid + ')')

    msg['Subject'] = "Your coupon from RecyclED"
    msg['From'] = "coupons@RecyclED.scot"
    msg['To'] = request.form['text']
    s.send_message(msg)
    return """
    <!doctype html>
    Email sent to: 
    """ + request.form['text']

def predict(file):
    return net.classify(file)


@app.route('/classify', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_loc)
            rv = predict(file_loc)
            os.remove(file_loc)
            return rv
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if args.debug:
    app.run(host='0.0.0.0')  # Run with Flask
else:
    serve(app, host='0.0.0.0', port=5000)  # Run with Waitress
