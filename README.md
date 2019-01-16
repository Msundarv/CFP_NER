# Call For Papers NER System (CFP_NER)

NER system that can tag people's names and affiliations on plain text 'call for papers'. Plain text 'call for papers' data is web scraped from [wikicfp](http://wikicfp.com) using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/). Names and affiliations that are associated with the 'call for papers' are identified using [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml) and [SpaCy](https://spacy.io/).

## Prerequisites

Requires Python 3.x.

Install [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for web scraping.

```bash
pip install beautifulsoup4
```
Install [NumPy](http://www.numpy.org/) and [NLTK](https://www.nltk.org/). NLTK is used for text pre-processing and for interfacing with Stanford NER.

```bash
pip install numpy
pip install nltk
```
Install [SpaCy](https://spacy.io/usage/).

```bash
pip install spacy
```

## How to use?
CFP_NER takes *'url'* and *'model'* as inputs. *'url'* is from where the 'Call For Papers' data have to be scraped from. Only 
 sub URLs to [wikicfp](http://wikicfp.com) is currently considered as valid URLs for this argument. *'model'* instructs CFR_NER System to use either SpaCy('m1') or Stanford NER('m2') to identify the names and affilliations associated with the 'Call For Papers'. *'url'* is a required argument whereas *'model'* is an optional argument. SpaCy('m1') is considered as default value for the argument *'model'*.

Usage: CFP_NER.py --url URL [--model {m1,m2}] [-h]

Sample usage
```bash
python CFP_NER.py --model "m2" --url "http://wikicfp.com/cfp/servlet/event.showcfp?eventid=72997"
```


## Author
[Sundar V](http://msundarv.com/) 