from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from fastai.vision.all import *

#Labeling function required for load_learner to work
def GetLabel(fileName):
  return fileName.split('-')[0]

learn = load_learner(Path('server/export.pkl')) #Import Model
app = Flask(__name__)
cors = CORS(app) #Request will get blocked otherwise on Localhost

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = PILImage.create(request.files['file'])
    # label,_,probs = learn.predict(img)
    # return f'{label} ({torch.max(probs).item()*100:.0f}%)'
    # lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    # img = PILImage.create(btn_upload.data[-1])
    # out_pl.clear_output()
    # with out_pl: display(img.to_thumb(128,128))
    score = learn.predict(img)
    attentiveness_score = 0
    for i in score[0]:
      if i == 'bending':
        attentiveness_score +=0.3
      elif i == 'chatting':
        attentiveness_score+=0.3
      elif i == 'raising hand':
        attentiveness_score+=0.9
      elif i == 'sitting':
        attentiveness_score+=0.5
      elif i == 'standing':
        attentiveness_score+=0.5
      else:
        attentiveness_score+=0.8
      
    avgScore = attentiveness_score/len(score[0])
    
    return f'Predictions: {score[0]} \n The Attentiveness Score for this Frame is: {avgScore:.01f}'


if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)



