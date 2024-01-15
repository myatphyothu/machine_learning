import os
import nltk
from core.currency_data import CurrencyData
from core.date_finder import DateFinder


home_dir = os.path.expanduser('~')
nltk_data_dir = f'{home_dir}/nltk_data'
nltk.data.path.append(nltk_data_dir)


class NLP(object):

    @staticmethod
    def _extract_amount(data, search_tags, trigger_words=[]):
        extracted_data = dict()
        data = [[w, t, False] for (w, t) in data]
        for i, (word, tag, extracted) in enumerate(data):

            if word.lower() in trigger_words:
                prev_pair = data[i - 1] if i != 0 else None
                next_pair = data[i + 1] if i + 1 < len(data) else None

                if prev_pair is not None and prev_pair[1] in search_tags and prev_pair[2] is False:
                    extracted_data[i + 1] = f'{word}{prev_pair[0]}'
                    data[i - 1][2] = True
                elif next_pair is not None and next_pair[1] in search_tags and next_pair[2] is False:
                    extracted_data[i + 1] = f'{word}{next_pair[0]}'
                    data[i + 1][2] = True

        return extracted_data

    @staticmethod
    def _clean_data_for_date_extraction(data):
        fn_map = {x: y for x, y in function_map.items() if x != 'date'}
        parsed = []
        for x, y in fn_map.items():
            parsed.extend(y(data))
        tokens = NLP.word_tokenize(data)
        return ' '.join(set(tokens) - set(parsed))

    @staticmethod
    def stop_words(language):
        return nltk.corpus.stopwords.words(language)

    @staticmethod
    def word_tokenize(data):
        return nltk.word_tokenize(data)

    @staticmethod
    def pos_tag(data):
        return nltk.pos_tag(nltk.word_tokenize(data))

    @staticmethod
    def create_named_entities(data):
        return nltk.ne_chunk(NLP.pos_tag(data))

    @staticmethod
    def extract_locations(data):
        entities = NLP.create_named_entities(data)
        locations = []
        for entity in entities:
            if hasattr(entity, 'label') and entity.label() == 'GPE':
                locations.append(' '.join(c[0] for c in entity.leaves()))
        return locations

    @staticmethod
    def extract_organizations(data):
        entities = NLP.create_named_entities(data)
        print(entities)
        organizations = []
        for subtree in entities.subtrees(filter=lambda t: t.label() == 'ORGANIZATION'):
            print(f'subtree: {subtree}')
            organization = ' '.join([leaf[0] for leaf in subtree.leaves()])
            organizations.append(organization)
        return organizations

    @staticmethod
    def extract_dates(data):
        dates = DateFinder.extract(data)
        return dates

    @staticmethod
    def extract_amounts(data):
        preprocessed = NLP.pos_tag(data)
        amounts = NLP._extract_amount(
            data=preprocessed,
            search_tags=['CD'],
            trigger_words=CurrencyData.currency_list + CurrencyData.symbols
        )
        print(f'amount: {amounts}')
        return amounts

    @staticmethod
    def extract(data, extract_list):
        output = dict()
        for item in extract_list:
            key = item.lower()
            if key in function_map:
                result = function_map[key](data)
                output[key] = result
            else:
                output[key] = 'Not implemented'
        return output

    # -------------------- SENTIMENT ANALYSIS -------------------------
    @staticmethod
    def freq_dist(tokenized_words):
        return nltk.FreqDist(tokenized_words)


function_map = {
    'date': NLP.extract_dates,
    'location': NLP.extract_locations,
    'amount': NLP.extract_amounts,
    'organization': NLP.extract_organizations
}