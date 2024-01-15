from flask import Flask, jsonify, request
from core.nlp import NLP
from core.sentiment_analyzer import SentimentAnalyzer as SA

app = Flask(__name__)

@app.route('/')
def default_service():
    return 'Welcome to Smart Lexi...'


@app.route('/word_tokenize', methods=['POST'])
def word_tokenize():
    input = request.get_json()['text']
    output = NLP.word_tokenize(input)
    return jsonify({'data': output})


@app.route('/pos_tag', methods=['POST'])
def pos_tag():
    input = request.get_json()['text']
    output = NLP.pos_tag(input)
    return jsonify({'data': output})


@app.route('/extract_locations', methods=['POST'])
def extract_locations():
    input = request.get_json()['text']
    output = NLP.extract_locations(input)
    return jsonify({'data': output})


@app.route('/extract_dates', methods=['POST'])
def extract_dates():
    input = request.get_json()['text']
    output = NLP.extract_dates(input)
    return jsonify({'data': output})


@app.route('/extract_amounts', methods=['POST'])
def extract_amounts():
    input = request.get_json()['text']
    print(input)
    output = NLP.extract_amounts(input)
    return jsonify({'data': output})


@app.route('/lexi_analysis', methods=['POST'])
def lexi_analysis():
    input = request.get_json()
    data = input['data']
    extract_list = input['keys']
    output = NLP.extract(data, extract_list)
    sentiment = SA.analyze(data)
    return jsonify({'data': output, 'sentiment': sentiment})
