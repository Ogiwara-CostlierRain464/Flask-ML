# Final Task for Script programming language class.
Yushi Ogiwara, #71970013

# Abstract
I made up a web application which can guess a number in a image.
I used Flask web application framework, and
I made up CNN(convolutional neural network), one of machine learning model, from scratch using numpy library.

# Usage
1. Run the following command, where <UPLOAD_FOLDER> specifies path to save uploaded images.

```shell script
python3 flask_front.py <UPLOAD_FOLDER>
```
 
2. Open http://127.0.0.1:5000 in web browser.
3. Press "Choose File" button, and select one image in `sample/` folder(this web application only accepts MNIST hand-written number image dataset).
4. Press Upload button. Then, web page jumps to http://127.0.0.1:5000/cnn
5. Wait for a while, and reload the page. Then, guessed number is printed.

Option: access http://127.0.0.1:5000/reset to reset state of application.

# Mechanism

## Machine Learning
CNN(convolutional neural network) can make local receptive fields easily, so it's tends to produce high 
accuracy for machine learning with images or videos. Now a days, CNN makes basics of deep neural network.
Not only that, thanks to back-propagation property, NN can treat it's layers as a module, so it's easy to implement.
Thus, I selected CNN to implement and use.

Code below, from CNN.py, explains my simple NN model.

```python
# Init layer
self.layers = OrderedDict()
self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
self.layers['Relu1'] = Relu()
self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
self.layers['Relu2'] = Relu()
self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

self.last_layer = SoftmaxWithLoss() 
```

At here, I don't do detailed explanation for CNN, but I'll explain
how Convolution layer can make local receptive fields, just a little.

![1](conv1.jpg)
Figure1: How convolution operation works (from [2]).

![2](conv2.jpg)
Figure2: Whole process of Convolution layer (from [2]).

Figure1 shows how convolution operation works and figure2 shows 
whole process of Convolution layer.

Image data have a special property, which is "Adjacent pixels have strong relationship", and which cannot 
retain in normal fully-connected NN.

As shown in figure 1, convolution operation can save this property.
Thus, this model tends to produce higher accuracy for machine learning with images.

Code below, from flask_front.py, shows how to use this model.

```python
img = np.array(Image.open(uploaded_file_path))
img = img.astype(np.float32)
img /= 255.0
img = img.reshape((1, 28, 28))

# 1
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 2
network.load_params("params.pkl")
# 3
result = network.predict(img[np.newaxis]).argmax(axis=1)
```

First, create instance of `SimpleConvNet` at #1,  
Second, I load trained network's weight & bias at #2,  
Finally, I predict number of `img` at #3.

## Front end
Compare to machine learning part, front end part is really simple.
I used Flask, which is micro web framework written in Python.

Code below, from flask_front.py, show how to handle image upload.

```python
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
            uploaded_file_path = os.path.join(upload_folder, filename)
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
```

When a browser access to the top page(with HTTP GET method),
it returns HTML code.
When a browser post image file(with HTTP POST method), 
application checks to filename, save it, and redirect to `/cnn`.

Code below, from flask_front.py, show how to handle uploaded image processing.

```python
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
```
When image file is uploaded, application creates image processing thread, and run it.
When the page is reloaded and image processing has done, application prints predicted number result.

# P.S.
I reverted NN train code(train.py & trainer.py)
I used AdaGrad algorithm for training.


# References
1. "Pattern Recognition and Machine Learning"
2. 「ゼロから作る Deep Learning」
