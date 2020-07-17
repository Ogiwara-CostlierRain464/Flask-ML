import os
from flask import Flask, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "gif"])


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
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

app.run()