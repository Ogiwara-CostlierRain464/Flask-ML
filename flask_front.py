import os
from flask import Flask, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import threading
from PIL import Image
from io import BytesIO
import numpy as np

from CNN import SimpleConvNet

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "gif"}


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

uploaded_file_path = None
ml_thread = None
result = None


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file")
            return redirect(request.url)
        if allowed_file(file.filename):
            global uploaded_file_path

            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(uploaded_file_path)

            return redirect("/cnn")
    return """
    <html>
    <head>
    <title>Upload file</title>
    </head>
    
    <body>
    <h1>Upload file please</h1>
    <form method=post enctype=multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    """


def cnn_predict():
    global result
    global ml_thread
    global uploaded_file_path

    img = np.array(Image.open(uploaded_file_path))
    img = img.astype(np.float32)
    img /= 255.0
    img = img.reshape((1, 28, 28))

    print(img)
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)

    network.load_params("params.pkl")
    result = network.predict(img[np.newaxis]).argmax(axis=1)


@app.route("/reset")
def reset():
    global result
    global ml_thread
    global uploaded_file_path

    result = None
    ml_thread = None
    uploaded_file_path = None
    return """
    Reset done.
    """


@app.route("/cnn")
def cnn():
    global result
    global ml_thread
    global uploaded_file_path

    if result is None:
        if ml_thread is None:
            ml_thread = threading.Thread(target=cnn_predict)
            ml_thread.start()
            return """ML started"""
        else:
            return """Now loading, please wait"""
    else:
        return f"result: {result}"


app.run()
