import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')


class TextPreprocessing:

    def __init__(self, filePath, rawText=False):
        self.stop_words = set(stopwords.words('english'))
        if (not rawText):

            file = open(filePath, "r", encoding='utf-8', errors='ignore')

            self.text = file.read()
        else:
            self.text = filePath
        text = ""
        for file_id in gutenberg.fileids():
            text += gutenberg.raw(file_id)

        # trainer = PunktTrainer()
        # trainer.INCLUDE_ALL_COLLOCS = True
        # trainer.train(text)

        # self.tokenizer = PunktSentenceTokenizer(trainer.get_params())
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def tokenize(self, text=None):
        if (text == None):
            return nltk.word_tokenize(self.text)
        else:
            return nltk.word_tokenize(text)

    def filter_sentecnce(self, text=None):
        if (text == None):
            words = nltk.word_tokenize(self.text)
        else:
            words = nltk.word_tokenize(self.text)
        return [w for w in words if not w in self.stop_words]

    def lemm(self, text=None):
        if (text == None):
            filtered_sentence = self.filter_sentecnce()
        else:
            filtered_sentence = self.filter_sentecnce(text)

        lem = [self.wordnet_lemmatizer.lemmatize(w) for w in filtered_sentence]
        return nltk.pos_tag(lem)

    def tokenize_sentences(self, text=None):
        if (text == None):
            return sent_tokenize(self.text)
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

    def remove_sentence(self, sentence):
        """
        Remove all sentences that have special patterns
        :return:
        """
        emails = '[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+'
        pipelines = '[A-Za-z0-9]+([ ]*[|][ ]*)+[A-Za-z0-9]+'
        paths = '[A-Za-z0-9]+([ ]*[/][ ]*)+[A-Za-z0-9]+'
        websites = '(www.)[a-zA-Z0-9]+'

        sentence = re.sub(emails, '', sentence)
        sentence = re.sub(pipelines, '', sentence)
        sentence = re.sub(paths, '', sentence)
        sentence = re.sub(websites, '', sentence)

        whiteList = '((?![A-Za-z0-9\s,;:\?\!\.\'"–%]).)*'
        sentence = re.sub(whiteList, '', sentence)

        return sentence

    def extract_cleaned_sentences(self):
        """
        Tokenize and return cleaned sentences
        :param paper_file_path:
        :return:
        """
        # Extract sentences
        if (self.text == None):
            raise Exception("File is not passed to text parser")
            return False

        sentences = self.tokenize_sentences()

        cleaned_sentences = []
        for sentence in sentences:
            # remove new lines
            sentence = self.cleanSentence(sentence)

            # if self.remove_sentence(sentence):
            #    continue

            cleaned_sentences.append(self.remove_sentence(sentence))
        return cleaned_sentences

    def stem_sentence(self, sentence):
        return self.stemmer.stem(sentence)

    def stem_verb(self, verb):
        return self.lemmatizer.lemmatize(verb, 'v')



