import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

sentences = [
    ('The quick brown fox jumps over the lazy dog.', 'A cat sits on the lazy dog.'),
    ('John gives Mary a gift', 'Mary receives a gift from John')
]


def similarity_score(s1, s2):
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)

    # find synonyms
    synsets1 = [wordnet.synsets(token) for token in tokens1]
    synsets2 = [wordnet.synsets(token) for token in tokens2]

    # compute similarity
    similarity_scores = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            tmp_similarity_scores = []
            for s1 in synset1:
                for s2 in synset2:

                    wup_score = s1.wup_similarity(s2)
                    if wup_score is not None:
                        tmp_similarity_scores.append(wup_score)

            if len(tmp_similarity_scores) > 0:
                max_score = max(tmp_similarity_scores)
                # print(f'max_score: {max_score}')
                similarity_scores.append(max_score)


            # similarity_scores.append(max([s1.wup_similarity(s2) for s1 in synset1 for s2 in synset2 if s1.wup_similarity(s2) is not None]))
    print('==== similarity scores')
    print(similarity_scores)
    x = sum(similarity_scores)
    y = len(similarity_scores)

    avg_similarity = sum(similarity_scores) / float(len(similarity_scores))
    print(f'sum: {x}, len: {y}, avg: {avg_similarity}')

    return avg_similarity


def analyze_similarity_score(score):
    if score > 0.5:
        print('Two sentences have the meaning.')
    else:
        print('Two sentences do not have the same meaning.')


def compare_sentences():
    for (s1, s2) in sentences:
        score = similarity_score(s1, s2)
        print(f'sentence1: {s1}')
        print(f'sentence2: {s2}')
        print(analyze_similarity_score(score))


if __name__ == '__main__':
    compare_sentences()
