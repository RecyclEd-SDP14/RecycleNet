import argparse

import flask
from flask import Flask, flash, request, redirect
from flask_mail import Mail, Message
from utils import delimiter
import flasknet
import os
import secrets
from werkzeug.utils import secure_filename
from waitress import serve
import db


parser = argparse.ArgumentParser(description='RecycleNet server runner')
parser.add_argument('--debug', action='store_true', help="Use dev server")
parser.add_argument('--new', action='store_true', help="Use new classification")
args = parser.parse_args()
if args.new:
    folder = "NewData"
else:
    folder = "OldData"


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), UPLOAD_FOLDER)
app.config['SECRET_KEY'] = secrets.token_hex(256)
app.config['MAIL_SERVER'] = 'smtp.sendgrid.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'apikey'
app.config['MAIL_PASSWORD'] = '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'
app.config['MAIL_DEFAULT_SENDER'] = 'RecyclED.scot@protonmail.com'
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

    msg = Message('Your coupon from Recycled', recipients=[request.form['text']], sender='RecyclED.scot@protonmail.com')
    msg.body = 'Thanks for using RecyclED. This coupon is totes worth £0.10. (Transaction id: ' + uid + ')'
    msg.html = 'Thanks for using RecyclED. This coupon is totes worth £0.10. (Transaction id: ' + uid + ')'
    mail = Mail(app)
    mail.send(msg)
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
            pr_class, p = predict(file_loc)
            os.remove(file_loc)
            rv = flask.jsonify(
                prediction=pr_class,
                confidence=float(p)
            )
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

@app.route('/stats')
def getstats():
    device_name = request.args.get('device')
    value = db.print_refund_value_per_device(device_name)
    items = db.print_recycling_totals_per_device(device_name)
    total = items[1]+items[2]+items[3]+items[4]
    response = flask.jsonify(
        value = int(value[1]),
        glass  = int(items[1]),
        plastic = int(items[2]),
        cans = int(items[3]),
        trash = int(items[4]),
        total = int(total)
    )
    return response



if args.debug:
    app.run(host='0.0.0.0')  # Run with Flask
else:
    serve(app, host='0.0.0.0', port=5000)  # Run with Waitress
