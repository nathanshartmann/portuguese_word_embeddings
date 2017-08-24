
"""
This script evaluates a embedding model in a semantic similarity perspective.
It uses the dataset of ASSIN sentence similarity shared task and the method
of Hartmann which achieved the best results in the competition.

ASSIN shared-task website:
http://propor2016.di.fc.ul.pt/?page_id=381

Paper of Hartmann can be found at:
http://www.linguamatica.com/index.php/linguamatica/article/download/v8n2-6/365
"""

from sklearn.linear_model import LinearRegression
from sentence_similarity.utils.assin_eval import read_xml, eval_similarity
from gensim.models import KeyedVectors
from xml.dom import minidom
from numpy import array
from os import path
import pickle
import argparse

DATA_DIR = 'sentence_similarity/data/'
TEST_DIR = path.join(DATA_DIR, 'assin-test-gold/')


def gensim_embedding_difference(data, field1, field2):
    """Calculate the similarity between the sum of all embeddings."""
    distances = []
    for pair in data:
        e1 = [i if i in embeddings else 'unk' for i in pair[field1]]
        e2 = [i if i in embeddings else 'unk' for i in pair[field2]]
        distances.append([embeddings.n_similarity(e1, e2)])
    return distances


def evaluate_testset(x, y, test):
    """Docstring."""
    l_reg = LinearRegression()
    l_reg.fit(x, y)
    test_predict = l_reg.predict(test)
    return test_predict


def write_xml(filename, pred):
    """Docstring."""
    with open(filename) as fp:
        xml = minidom.parse(fp)
    pairs = xml.getElementsByTagName('pair')
    for pair in pairs:
        pair.setAttribute('similarity', str(pred[pairs.index(pair)]))
    with open(filename, 'w') as fp:
        fp.write(xml.toxml())


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Sentence similarity evaluation for word embeddings in
        brazilian and european variants of Portuguese language. It is expected
        a word embedding model in text format.''')

    parser.add_argument('embedding',
                        type=str,
                        help='embedding model')

    parser.add_argument('lang',
                        choices=['br', 'eu'],
                        help='{br, eu} choose PT-BR or PT-EU testset')

    args = parser.parse_args()
    lang = args.lang
    emb = args.embedding

    # Loading embedding model
    embeddings = KeyedVectors.load_word2vec_format(emb,
                                                   binary=False,
                                                   unicode_errors="ignore")

    # Loading evaluation data and parsing it
    with open('%sassin-pt%s-train.pkl' % (DATA_DIR, lang), 'rb') as fp:
        data = pickle.load(fp)

    with open('%sassin-pt%s-test-gold.pkl' % (DATA_DIR, lang), 'rb') as fp:
        test = pickle.load(fp)

    # Getting features
    features = gensim_embedding_difference(data, 'tokens_t1', 'tokens_t2')
    features_test = gensim_embedding_difference(test, 'tokens_t1', 'tokens_t2')

    # Predicting similarities
    results = array([float(i['result']) for i in data])
    results_test = evaluate_testset(features, results, features_test)

    write_xml('%soutput.xml' % DATA_DIR, results_test)

    # Evaluating
    pairs_gold = read_xml('%sassin-pt%s-test.xml' % (TEST_DIR, lang), True)
    pairs_sys = read_xml('%soutput.xml' % DATA_DIR, True)
    eval_similarity(pairs_gold, pairs_sys)
