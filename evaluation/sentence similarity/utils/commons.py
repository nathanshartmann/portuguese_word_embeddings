# -*- coding: utf-8 -*-

"""
Common structures and functions used by other scripts.
"""

from xml.etree import cElementTree as ET


str_to_entailment = {'none': 0,
                     'entailment': 1,
                     'paraphrase': 2}
entailment_to_str = {v: k for k, v in str_to_entailment.items()}

class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, t, h, id_, entailment, similarity):
        '''
        :param t: string with the text
        :param h: string with the hypothesis
        :param id_: int indicating id in the original file
        :param entailment: int indicating entailment class
        :param similarity: float
        '''
        self.t = t
        self.h = h
        self.id = id_
        self.entailment = entailment
        self.similarity = similarity


def read_xml(filename, need_labels):
    '''
    Read an RTE XML file and return a list of Pair objects.
    :param filename: name of the file to read
    :param need_labels: boolean indicating if labels should be present
    '''
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()

    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        id_ = int(attribs['id'])

        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()

            try:
                ent_value = str_to_entailment[ent_string]
            except ValueError:
                msg = 'Unexpected value for attribute "entailment" at pair {}: {}'
                raise ValueError(msg.format(id_, ent_string))

        else:
            ent_value = None

        if 'similarity' in attribs:
            similarity = float(attribs['similarity'])
        else:
            similarity = None

        if need_labels and similarity is None and ent_value is None:
            msg = 'Missing both entailment and similarity values for pair {}'.format(id_)
            raise ValueError(msg)

        pair = Pair(t, h, id_, ent_value, similarity)
        pairs.append(pair)

    return pairs
