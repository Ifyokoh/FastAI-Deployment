from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from fastai.vision.all import *
import cv2

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
    upload = request.files['file']
    if upload.lower().endswith('.png'):
      img = PILImage.create(upload)
      # return f'{label} ({torch.max(probs).item()*100:.0f}%)'
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
      
      return f"Predictions: {score[0]}\nThe Attentiveness Score for this Frame is: {avgScore:.01f}"

    elif upload.lower().endswith('.mp4'):
        for fn in upload.keys():
          vidcap = cv2.VideoCapture(fn)
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite("image"+str(count)+".png", image)     # save frame as JPG file
            return hasFrames
        sec = 0
        frameRate = 1 #//it will capture image in each 0.5 second
        count=1
        success = getFrame(sec)
        while success:
            count = count + 5
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)

        # score = learn_inf.predict(img)
        global df
        dataframe = []
        for frame in glob.iglob('*.png'):
          # print(learn_inf.predict(frame))
          value = learn_inf.predict(frame)
          # print(value[0])


          attentiveness_score = 0
          for i in value[0]:
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
          try:
            avgScore = round(attentiveness_score/len(value[0]), 1)
          except:
            pass
          dataframe.append([frame, value[0], avgScore])

        pd.set_option('display.max_colwidth', -1)
        df = pd.DataFrame(dataframe, columns=["frame", "predicted_classes", "attentiveness_score"])
        # lbl_pred.value = f'{pd.DataFrame(dataframe, columns=["frame", "predicted_classes", "attentiveness_score"])}'
        print(df)
    
    else:
      print('Upload a video or an Image')


if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)



