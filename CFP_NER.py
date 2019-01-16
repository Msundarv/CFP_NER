"""Call For Papers NER System

This NER system can tag people's names and affiliations on plain text 'call for papers'.
Plain text 'call for papers' data is web scraped from http://wikicfp.com using Beautiful Soup.
Names and affiliations that are associated with the 'call for papers' are identified using Stanford NER and SpaCy.
"""

import os
import argparse
import requests
import en_core_web_sm
import nltk
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags


def get_cfp(url: str) -> str:
    """Does the website scraping on given URL and returns the 'Call For Papers' section.
    Along with scraping, this function does some text pre processioning.
    This function's web scraping currently supports only http://wikicfp.com

    :param url: URL string to scrape
    :type url: str
    :return: Scraped and preprocessed 'call for papers' string
    :rtype: str
    """

    try:
        html = requests.get(url).text  # Storing the server's response for the HTTP GET request
        soup = BeautifulSoup(html, 'html5lib')  # Uses Beautiful Soup to scrape the 'call for papers' data
        cfp = soup.find('div', attrs={'class': 'cfp'}).find_all(text=True)
        cfp = ' '.join(cfp)
        cfp = cfp.encode('ASCII', 'ignore').decode('ASCII')  # To remove the special characters and symbols
        cfp = cfp.strip().replace('\n', ' ').replace('\t', ' ')
    except AttributeError as error:  # Exception caused by invalid URLs like http://wikicfp.com
        print('Please enter a valid URL')
        exit(0)

    return cfp


def spacy_ner(cfp: str) -> list:
    """This function detects and classifies the names and affiliations associated with the given 'call for papers' string.
    Above mentioned detection and classification is done using SpaCy's NER System.

    :param cfp: Scraped and preprocessed 'call for papers' string
    :type cfp: str
    :return: List of named entity along with their category('PERSON' or 'ORG')
    :rtype: list
    """

    nlp = en_core_web_sm.load()
    ne_chunks = nlp(cfp).ents  # Tuple of named entity as SpaCy's Span objects

    # Filtering only 'ORG' and 'PERSON' entity as well as converting the return type to list of list
    result = []
    for chunk in ne_chunks:
        if chunk.label_ == 'ORG' or chunk.label_ == 'PERSON':
            result.append([chunk.text.strip(), chunk.label_])
    return result


stanford_ner_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), './Lib/stanford-ner-2018-10-16')


def stanford_ner(cfp: str) -> list:
    """This function detects and classifies the names and affiliations associated with the given 'call for papers' string.
    Above mentioned detection and classification is done using Stanford NER System.
    NLTK Library's interface for Stanford NER System is used.

    :param cfp: Scraped and preprocessed 'call for papers' string
    :type cfp: str
    :return: List of named entity along with their category('PERSON' or 'ORG')
    :rtype: list
    """

    # Loading Stanford NER Tagger
    st = StanfordNERTagger(os.path.join(stanford_ner_path, './classifiers/english.muc.7class.distsim.crf.ser.gz'),
                           os.path.join(stanford_ner_path, './stanford-ner.jar'),
                           encoding='utf-8')
    ne_tokenchunks = st.tag(word_tokenize(cfp))  # List of token level named entity tuples

    # Using IOB Tagging to group continuous sequential named entities of same type
    # Filtering only 'ORG' and 'PERSON' entity as well as converting the return type to list of list
    ne_parser = nltk.RegexpParser('Tag: {<ORGANIZATION|PERSON.*>+}')
    ne_chunks = tree2conlltags(ne_parser.parse(ne_tokenchunks))
    result = []
    for i, chunk in enumerate(ne_chunks):
        if chunk[2] == 'B-Tag':
            ent_type = chunk[1]
            ent = chunk[0].strip()
            for chunkseq in ne_chunks[i+1:]:  # Merges all I-Tags following the current B-Tag to form a new named entity
                if chunkseq[2] == 'I-Tag':
                    ent += ' '+chunkseq[0].strip()
                if chunkseq[2] != 'I-Tag':
                    break
            if ent_type == 'ORGANIZATION':  # 'ORGANIZATION' category given by Stanford NER is converted to 'ORG'
                ent_type = 'ORG'
            result.append([ent, ent_type])
    return result


if __name__ == '__main__':

    # Step 1: Parsing the input arguments
    parser = argparse.ArgumentParser(description='To tag people\'s names and affiliations on \'Call For Papers\'',
                                     epilog='',
                                     add_help=False)
    group1 = parser.add_argument_group('required arguments')
    group1.add_argument('--url', type=str, required=True,
                        help='URL from where the \'Call For Papers\' data is scraped from.')
    group2 = parser.add_argument_group('optional arguments')
    group2.add_argument('--model', type=str, default='m1', choices=['m1', 'm2'],
                        help='NER System used to identify the names and affiliations.'
                             '\n\'m1\' - SpaCy NER(Default)'
                             '\n\'m2\' - Stanford NER')
    group2.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit')
    # print(parser.print_help())
    args = parser.parse_args()
    # print(args)

    # Step 2: Web scraping and text pre processing
    # Check for URL validity before continuing
    # TODO: All sub URLs to http://wikicfp.com is considered valid now which can be improved
    url = urlparse(args.url)
    if 'wikicfp.com' not in url:
        print('Please enter a valid URL')
        exit(0)
    # print(url.geturl())
    cfp = get_cfp(url.geturl())
    # print(cfp)

    # Step 3: Find the names and affiliations associated with the 'call for papers' using either SpaCy or Stanford NER
    result = []
    if args.model == 'm1':
        # Default model when the argument is not mentioned by the user
        result = spacy_ner(cfp)
    else:
        result = stanford_ner(cfp)
    # print(result)
    # TODO: Result can be improved further by merging SpaCy and Stanford NER

    # Step 4: Output the results
    names = []
    affiliations = []
    for entity in result:
        if entity[1] == 'ORG':
            affiliations.append(entity[0])
        if entity[1] == 'PERSON':
            names.append(entity[0])
    names.sort()
    affiliations.sort()
    print('***Names***\n', ' * '.join(names), '\n')
    print('***Affiliations***\n', ' * '.join(affiliations))
