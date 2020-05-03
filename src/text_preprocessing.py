import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
import re
import unicodedata
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')


class TextPreprocessing:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
            return nltk.word_tokenize(text)

    def filter_sentecnce(self, text):
        if (text == None):
            words = nltk.word_tokenize(self.text)
        else:
            words = nltk.word_tokenize(self.text)
        return [w for w in words if not w in self.stop_words]

    def lemm(self, text):
        filtered_sentence = self.filter_sentecnce(text)
        lem = [self.wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence]
        return nltk.pos_tag(lem)

    def tokenize_sentences(self, text):
        return sent_tokenize(text)

    def cleanSentence(self, sentence):
        """
        Remove newlines from a sentence
        :param sentence:
        :return: cleaned sentence
        """

        # accounts for one word that spans 2 lines
        sentence = sentence.replace("-\n", "")

        # account for simple new line
        cleaned_sentence = sentence.replace("\n", " ")

        # remove hyphens
        cleaned_sentence = cleaned_sentence.replace("-", " ")

        return cleaned_sentence

    def remove_whiteList(self, sentence):
        """
        Remove all sentences that have special patterns
        :return:
        """
        emails = '[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+'
        websites = '(http[s]*:[/][/])[a-zA-Z0-9]+'
        mentions = '@[A-Za-z0-9]+'
        sentence = re.sub(emails, '', sentence)
        sentence = re.sub('#', '', sentence)
        sentence = re.sub(websites, '', sentence)
        sentence = re.sub(mentions, '', sentence)
        whiteList = '((?![A-Za-z0-9\s,;:\?\!\.\'"–%]).)*'
        sentence = re.sub(whiteList, '', sentence)

        return sentence

    def stem_sentence(self, sentence):
        return self.stemmer.stem(sentence)

    def stem_verb(self, verb):
        return self.lemmatizer.lemmatize(verb, 'v')

    def remove_accented_chars(self,text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(self,text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text




