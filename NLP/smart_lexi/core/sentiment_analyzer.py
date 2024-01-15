from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer(object):

    sia = SentimentIntensityAnalyzer()

    @staticmethod
    def analyze(text):
        polarity_scores = SentimentAnalyzer.sia.polarity_scores(text)
        return {
            'polarity scores': polarity_scores,
            'is positive': polarity_scores['compound'] > 0
        }
    