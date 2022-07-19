from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import stopwords
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from lxml import etree
import logging
import pandas as pd

# nltk requirements, comment out after first run
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
xml_path = 'news.xml'
en_exceptions = list(stopwords.words('english'))


def remove_punkt(token_list):
    # Removes punctuation from the given list
    tokens_no_punkt = []
    for word in token_list:
        if word not in list(string.punctuation):
            tokens_no_punkt.append(word)
    return tokens_no_punkt


def tokenizer(text):
    return tokenize.word_tokenize(text.lower())


def filter_pos_tags(token_list, pos_tag='NN'):
    # Returns a list with words that have the tag in pos_tag, defaults to NN (noun)
    filtered_list = []
    for token in token_list:
        if nltk.pos_tag([token])[0][1] == pos_tag:
            filtered_list.append(token)
    return filtered_list


def remove_stops(token_list):
    # Removes all stop words defined in en: currently only stopwords.word (en)
    token_list0 = []
    for token in token_list:
        if token not in en_exceptions:
            token_list0.append(token)
    return token_list0


def lemmatize(token_list):
    # Lemmatize the given token list with the WordNetLemmatizer
    # Information on WNL here: https://hyperskill.org/learn/step/10451
    wordnet = WordNetLemmatizer()
    for ind, token in enumerate(token_list):
        token_list[ind] = wordnet.lemmatize(token)
    return token_list


def parse_articles(xml_file):
    # Reads articles from a xml file, returns a dict {title: article_body}
    tree = etree.parse(xml_file)
    root = tree.getroot()[0]
    articles = dict()
    for news in root:
        for el in news:
            if el.get('name') == 'head':
                title = el.text
                articles[title] = ''
            elif el.get('name') == 'text':
                articles[title] = el.text.replace('\\n', '')
    return articles


def get_tfidf(dataset):
    # Transforms the given dataset to a weighted matrix
    vectorizer = TfidfVectorizer()
    # input='content', use_idf=True, lowercase=False, analyzer='word', ngram_range=(1, 1),
    # stop_words=None, vocabulary=None, min_df=0.01, max_df=0.60
    weighted_matrix = vectorizer.fit_transform(dataset)
    global terms
    terms = vectorizer.get_feature_names_out()
    return weighted_matrix


articles = parse_articles(xml_path)
article_tokens = dict()
dataset = []

for title, article in articles.items():  # Build the dataset from dict
    string_to_vectorize = ''
    # The two lines below could be combined to reduce memory usage, but further reduct readability
    article_tokens[title] = filter_pos_tags(remove_punkt(remove_stops(lemmatize(tokenizer(article.lower())))))
    dataset.append(' '.join(article_tokens[title]))

matrix = get_tfidf(dataset)

for i, article_name in enumerate(article_tokens.keys()):  # Sorting the results and presentation
    print(article_name + ':')
    data = []
    for col, term in enumerate(terms):
        if matrix[i, col] > 0.1:  # Reducing results to help readability
            data.append((term, matrix[i, col]))
    # logging.debug(data)
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking = ranking.sort_values(['rank', 'term'], ascending=[False, False])
    logging.debug(ranking)
    for j in range(5):
        print(ranking.iloc[j]['term'], end=' ')
    if i < len(article_tokens.keys()) - 1:
        print()
        print()
