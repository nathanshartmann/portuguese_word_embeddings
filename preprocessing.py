
"""
Script used for cleaning corpus in order to train word embeddings.

All emails are mapped to a EMAIL token.
All numbers are mapped to 0 token.
All urls are mapped to URL token.
Different quotes are standardized.
Different hiphen are standardized.
HTML strings are removed.
All text between brackets are removed.
All sentences shorter than 5 tokens were removed.
...
"""

from sys import stdout
import argparse
import re
import nltk

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

# Punctuation list
punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

# ##### #
# Regex #
# ##### #
re_remove_brackets = re.compile(r'\{.*\}')
re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
re_transform_numbers = re.compile(r'\d', re.UNICODE)
re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
re_tree_dots = re.compile(u'…', re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                       (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                         (punctuations, punctuations), re.UNICODE)
re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
re_changehyphen = re.compile(u'–')
re_doublequotes_1 = re.compile(r'(\"\")')
re_doublequotes_2 = re.compile(r'(\'\')')
re_trim = re.compile(r' +', re.UNICODE)


def clean_text(text):
    """Apply all regex above to a given string."""
    text = text.lower()
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used for cleaning corpus in order to train
        word embeddings.''')

    parser.add_argument('input',
                        type=str,
                        help='input text file to be cleaned')

    parser.add_argument('output',
                        type=str,
                        help='output text file')

    args = parser.parse_args()
    f_in = args.input
    f_out = args.output

    txt, wc_l = [], 0
    final = []
    with open(f_in, 'r', encoding='utf8') as f:
        wc_l = sum(1 for l in f)

    # Clean lines.
    with open(f_in, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            stdout.write('Reading lines...')
            stdout.write('%8d/%8d \r' % (i + 1, wc_l))
            stdout.flush()
            txt.append(clean_text(line))

    # Tokenize and remove short and malformed sentences.
    for line in txt:
        for sent in sent_tokenizer.tokenize(line):
            if sent.count(' ') >= 3 and sent[-1] in ['.', '!', '?', ';']:
                if sent[0:2] == '- ':
                    sent = sent[2:]
                elif sent[0] == ' ' or sent[0] == '-':
                    sent = sent[1:]
                final.append(sent)

    vocab, tokens = set(), 0
    with open(f_out, 'w', encoding='utf8') as fp:
        for sent in final:
            fp.write('%s\n' % sent)
            tokens += sent.count(' ') + 1
            for w in sent.split():
                vocab.add(w)

    print('Tokens: ', tokens)
    print('Vocabulary: ', len(vocab))
