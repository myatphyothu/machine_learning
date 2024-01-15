import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize




def stop_words_stuff():
    print('------ STOP WORDS STUFF -------')
    sentence = """He said the latest object "violated Canadian airspace" and was shot down over Yukon in north-west Canada."""
    stop_words = set(stopwords.words('english'))
    # print('======================================== STOP WORDS (ENGLISH) =============================================')
    # print(f'There are {len(stop_words)} stopwords(english).')
    # print(stop_words)
    words = word_tokenize(sentence)
    not_stopwords = [w for w in words if not w in stop_words]
    print(sentence)
    print('> Filtered:')
    print(not_stopwords)


def stemming_stuff():
    from nltk.stem import PorterStemmer

    sentence = """The American military destroyed a Chinese balloon last weekend, 
    and on Friday an unspecified object the size of a small car was shot down off Alaska
    """

    print('------- STEMMING STUFF -------')
    words = word_tokenize(sentence)

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    print(sentence)
    print('> Stemmed:')
    print(stemmed)


def lemmatizing_stuff():
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    sentence = """The American military destroyed a Chinese balloon last weekend, 
        and on Friday an unspecified object the size of a small car was shot down off Alaska
        """
    print('------- LEMMING STUFF -------')
    lemmer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmed_words = [lemmer.lemmatize(w, pos=wordnet.VERB) for w in words]
    lemmed_sentence = ' '.join(lemmed_words)
    print(f'original: {sentence}')
    print(f'lemmed: {lemmed_sentence}')


if __name__ == '__main__':
    # stop_words_stuff()
    # stemming_stuff()
    lemmatizing_stuff()

