from flask import Flask,request,render_template,session, redirect, url_for
import pickle
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = load_model('C:/Users/odhia/OneDrive/Desktop/streamlit tut/spotifyReviewModel.keras')
tokenizer = pickle.load(open('spotifytokenizer.pkl','rb'))

songs = pickle.load(open('tracks.pkl','rb'))
sim = pickle.load(open('cosine.pkl','rb'))

track_names = songs['Track Name'].values
artists = songs['Artist Name(s)'].values

def reccomend(new_song):
    ind = songs[songs['Track Name'] == new_song].index[0]
    distance = sorted(list(enumerate(sim[ind])),reverse = True,key = lambda x: x[1])
    recommendations = []
    for i in distance[1:6]:
        recommendations.append(f"{songs['Track Name'].iloc[i[0]]} by {songs['Artist Name(s)'].iloc[i[0]]}")
        
    return recommendations

def preprocess_text(sentence):
    tag_pattern = re.compile(r'<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Lowercasing
    sent = sentence.lower()
    # Removal of HTML Tags
    sent = re.sub(tag_pattern, '', sent)
    # Removing Punctuation & Special Characters
    sent = re.sub('[^a-zA-Z]',' ',sent)
    # removing single character
    sent = re.sub(r"\s+[a-zA-Z]\s+",' ',sent)
    # removing multiple spaces
    sent = re.sub(r'\s+',' ',sent)
    # Removal of URLs
    sent = re.sub(url_pattern,'',sent)
    return sent
                               
@app.route('/')
def homepage():
    return render_template('spotify.html',track_names=track_names, reccos=session.get('reccos'), text=session.get('text'), song_title=session.get('song_title'))

@app.route('/recomend',methods = ['POST'])
def do_recommendation():
    song_title = request.form.get('trackSelect')
    reccos = reccomend(song_title)
    session['reccos'] = reccos
    session['song_title'] = song_title
    return redirect(url_for('homepage'))

@app.route('/analyse',methods = ['POST'])
def do_analysis():
    corpus = request.form.get('review')
    inp = []
    inp.append(preprocess_text(corpus))
    inp = tokenizer.texts_to_sequences(inp)
    inp = pad_sequences(inp,padding='pre',maxlen = 100)
    
    pred = model.predict(inp)
    if pred > 0.5:
        msg = 'Thank you for the positive feedback,enjoy your music'
    else:
        msg = "Well,we'll consider that and improve on it,enjoy your music if you don't mind"
    session['text'] = msg
    return redirect(url_for('homepage'))


if __name__ == '__main__':
    app.run(host="0.0.0.0")