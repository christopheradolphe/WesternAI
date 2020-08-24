!pip install flask_wtf
from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField, FileField
from wtforms.validators import NumberRange
from markupsafe import escape

import numpy as np  
import joblib

def return_prediction(model,image):
    
    output = model(image)

    predicted_activasions, predicted_class = pt.max(output,1)
    
    if predicted_class == 0:
      return 'Mild Demented'
    if predicted_class == 1:
      return 'Moderately Demented'
    if predicted_class == 2:
      return 'Non Demented'
    if predicted_class == 3:
      return 'Very Mildly Demented'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

alzeimers_model = pt.load('/content/checkpoints/final_alzeimers_model.h5')
model.load_state_dict(pt.load('/content/checkpoints/mnist-009.pkl'))

class ImageForm(FlaskForm):
    image = FileField('Image:')

    submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = ImageForm()

    if form.validate_on_submit():

        session['image'] = form.image.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():

    image = 'image'

    image_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.Grayscale(1), #making the image gray scale so there is only one channel
                                        transforms.ToTensor()])
    
    image = image_transforms(image)
    


    image = pt.utils.data.DataLoader(image)
    

    results = return_prediction(model=alzeimers_model,image=image)

    return render_template('prediction.html',results=results)
"""
@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)
"""

@app.route('/next_steps')
def next_steps():
  return 'These are the next steps to take'

@app.route('/about')
def about():
    return 'The about page'

if __name__ == '__main__':
    export FLASK_APP = alzheimers.py
    flask run --host=0.0.0.0
