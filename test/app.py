import gensim
import joblib
from flask import Flask, render_template, request

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__, template_folder='/home/hnhan/Downloads/DEMO_ML_FLASK/template')

class PredictTitle:
  def __init__(self, model_file = 'LSVC.sav'):
    self.model = joblib.load(model_file)
  
  def predict(self, input_title):

    X_train = joblib.load('/home/hnhan/Downloads/DEMO_ML_FLASK/X_train (2).sav')
    X_train = np.array(X_train)
    lines = input_title
    lines = gensim.utils.simple_preprocess(lines)
    lines = ' '.join(lines) 
    lines = ViTokenizer.tokenize(lines)
    lines = ''.join(lines)
    count_vect = TfidfVectorizer(analyzer='word', max_features=100000)
    lines = [lines]
    count_vect.fit(X_train)
    lines = count_vect.transform(lines)
    #return lines
    title_predict = self.model.predict(lines)
    return title_predict

pt = PredictTitle()

@app.route('/')
def ping():
    return 'ok'

@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'GET':
        return render_template('full.html')
    
    else:
    
        title = request.form['title_result']
        prediction = pt.predict(title)
        if (prediction==0):
            class_title='Thuộc chủ đề chính trị'
        if (prediction==1):
            class_title='Thuộc chủ đề công nghệ'
        if (prediction==2):
            class_title='Thuộc chủ đề giáo dục'
        if (prediction==3):
            class_title='Thuộc chủ đề kinh doanh'
        if (prediction==4):
            class_title='Thuộc chủ đề pháp luật'
        if (prediction==5):
            class_title='Thuộc chủ đề thể thao'
        return render_template('full.html', class_title=class_title)
        
 
if __name__ == '__main__':
    app.run(debug=True)

