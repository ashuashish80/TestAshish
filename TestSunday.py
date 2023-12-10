from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np

app = Flask(__name__)
api = Api(app)

model = tf.keras.models.load_model('sentiment_model.h5')

from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/db_name'
db = SQLAlchemy(app)

class SentimentPrediction(Resource):
    def post(self):
        data = request.get_json(force=True)
        text = data['text']

        text_sequence = pad_sequences([[word_index[word] for word in text.split() if word in word_index]], padding='post', maxlen=maxlen)

        prediction = model.predict(np.array(text_sequence))

        log_entry = PredictionLog(text=text, prediction=float(prediction[0]))
        db.session.add(log_entry)
        db.session.commit()

        return {'prediction': float(prediction[0])}

api.add_resource(SentimentPrediction, '/predict')

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.Float, nullable=False)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
