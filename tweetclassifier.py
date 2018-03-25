import os
import csv
import sys
import config
import string
import pickle
import tarfile
import numpy as np
import pandas as pd
from collections import defaultdict
# from gensim import corpora
# from gensim.models import ldamodel, Phrases, phrases
# from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer


class classifytweet:
    def __init__(self, model_files='model_files/'):
        """
        tweet_body: the individual tweet
        model_files: the location for the model files. leave the model files folder as is
                     unless you have changed the location of all the model files.
        """
        self.model_files = model_files
        self.load_model()
        # self.n = len(self.dictionary.items())
        # self.word_map = {v:k for k,v in self.dictionary.items()}
        print("Model items loaded and classifier initialized!")

    def load_model(self):
        """
        Loads the model, corpus, and dictionary.
        """
        # extract tarfiles
        for f in os.listdir(self.model_files):
            if f.endswith('.gz'):
                tar = tarfile.open(self.model_files + f, "r:gz")
                tar.extractall(path=self.model_files)
                tar.close()

        # # load model, corpus, and dictionary objects
        # fnames = [fn for fn in os.listdir(self.model_files) if '.gensim' in fn]
        # self.model = ldamodel.LdaModel.load(self.model_files + fnames[0])
        # self.corpus = corpora.MmCorpus(self.model_files + 'unigrams_corpus.mm')
        # self.dictionary = corpora.Dictionary.load(self.model_files + 'unigrams_dictionary.pkl')
        # self.model.id2word = self.dictionary
        # self.phraser = phrases.Phrases.load(self.model_files + 'document_phraser.pkl')
        # for f in ['unigrams_dictionary.pkl', 'unigrams_corpus.mm', 'unigrams_corpus.mm.index', 'NB_vectorizer.pkl', 'NB_sentiment_model.pkl']:
        #     fnames.append(f)

        # load the valence and arousal arrays
        df = pd.read_csv(config.valence_arousal)
        words = df.Word.values
        valence_mean = df['V.Mean.Sum'].values
        valence_sd = df['V.SD.Sum'].values
        arousal_mean = df['A.Mean.Sum'].values
        arousal_sd = df['A.SD.Sum'].values
        self.val_ar = {}
        self.val_ar = defaultdict(lambda:{'valence':{'mean':0,'sd':0},'arousal':{'mean':0,'sd':0}}, self.val_ar)
        for ix,w in enumerate(words):
            # self.val_ar[w]={'valence':{'mean':0,'sd':0},'arousal':{'mean':0,'sd':0}}
            self.val_ar[w]['valence']['mean'] = valence_mean[ix]
            self.val_ar[w]['valence']['sd'] = valence_sd[ix]
            self.val_ar[w]['arousal']['mean'] = arousal_mean[ix]
            self.val_ar[w]['arousal']['sd'] = arousal_sd[ix]
        del df,words,valence_mean,valence_sd,arousal_mean,arousal_sd
        # self.valence_mean = pickle.load(open(self.model_files + 'valence_mean.pkl', 'rb'))
        # self.arousal_mean = pickle.load(open(self.model_files + 'arousal_mean.pkl', 'rb'))
        # self.valence_sd = pickle.load(open(self.model_files + 'valence_sd.pkl', 'rb'))
        # self.arousal_sd = pickle.load(open(self.model_files + 'arousal_sd.pkl', 'rb'))

        # load the MinMaxScaler for the transforming the scores
        # self.base_outrage_scaler = None
        # self.expanded_outrage_scaler = None
        # self.valence_scaler = None
        # self.arousal_scaler = None
        # self.emoji_scaler = None
        # self.topic_valence_scaler = pickle.load(open(self.model_files + 'topic_valence_scaled.pkl', 'rb'))
        # self.topic_arousal_scaler = pickle.load(open(self.model_files + 'topic_arousal_scaled.pkl', 'rb'))

        # load the Naive Bayes sentiment model
        try:
            self.nb_model = pickle.load(open(config.nb_model, 'rb'))
        except:
            self.nb_model = pickle.load(open(config.nb_model, 'rb'), encoding='latin1')
        try:
            self.nb_vectorizer = pickle.load(open(config.nb_vectorizer, 'rb'))
        except:
            self.nb_vectorizer = pickle.load(open(config.nb_vectorizer, 'rb'), encoding='latin1')

        # load the outrage dictionaries
        with open(config.outrage, 'r') as f:
            self.outrage_list = list(csv.reader(f))[0]
        with open(config.exp_outrage, 'r') as f:
            self.exp_outrage_list = list(csv.reader(f))[0]

        # cleanup the unzipped files
        # for f in fnames:
        #     os.remove(self.model_files + f)
        for f in os.listdir(self.model_files):
            if f.endswith('.gz'):
                os.remove(self.model_files + f[:-3])

        # set up the lists for preparing the tweet
        keep = set(['!','?'])
        self.stop = set(stopwords.words('english'))
        remove = set([x for x in list(string.punctuation) if x not in keep])
        self.stop.update(remove)
        self.stop.update(['',' ','  '])
        del keep,remove

    def prepare_tweet(self, tweet):
        """
        Turn that unstructured text into sweet, sweet, "cleaned" up tokens!
        """
        self.tweet = tweet
        stemmer = SnowballStemmer("english")
        tokenizer = TweetTokenizer()
        self.tweet_tokenized = tokenizer.tokenize(self.tweet)
        self.n = len(self.tweet_tokenized)
        try:
            self.tweet_tokenized = [unicode(y.encode("utf-8"), errors='ignore') for y in self.tweet_tokenized]
            self.stemmed = [stemmer.stem(y) for y in self.tweet_tokenized]
        except:
            self.stemmed = [stemmer.stem(y) for y in self.tweet_tokenized]

        self.stemmed = [d for d in self.stemmed if d not in self.stop]
        # self.phrased = list(self.phraser[[stemmed]])[0]

    def get_valence_score(self):
        """
        Creates the valence and arousal score for the tweet.
        """
        # tweet_arr = np.zeros(self.n)
        # for word in set(self.phrased) & set(self.word_map.keys()):
        #     tweet_arr[self.word_map[word]] = 1.
        # mean = tweet_arr * self.valence_mean
        # sd = tweet_arr * self.valence_sd
        tweet_arr = np.zeros(self.n)
        mean = np.zeros(self.n)
        sd = np.zeros(self.n)
        comp = set(self.stemmed) &  set([*self.val_ar])
        for ix,w in enumerate(self.stemmed):
            if w in comp:
                tweet_arr[ix] = 1.
                mean[ix] = self.val_ar[w]['valence']['mean']
                sd[ix] = self.val_ar[w]['valence']['sd']

        total_sd = np.sum(sd) * tweet_arr
        with np.errstate(divide='ignore', invalid='ignore'):
            sd_ratio = total_sd / sd
            sd_ratio[sd == 0] = 0
        sd_weight = sd_ratio / np.sum(sd_ratio)

        if np.sum(mean*sd_weight) == np.nan:
            self.valence_score = 0
        else:
            self.valence_score = np.sum(mean*sd_weight)

        return self.valence_score

    def get_arousal_score(self):
        """
        Creates the valence and arousal score for the tweet.
        """
        # tweet_arr = np.zeros(self.n)
        # for word in set(self.phrased) & set(self.word_map.keys()):
        #     tweet_arr[self.word_map[word]] = 1.
        # mean = tweet_arr * self.arousal_mean
        # sd = tweet_arr * self.arousal_sd
        tweet_arr = np.zeros(self.n)
        mean = np.zeros(self.n)
        sd = np.zeros(self.n)
        comp = set(self.stemmed) &  set([*self.val_ar])
        for ix,w in enumerate(self.stemmed):
            if w in comp:
                tweet_arr[ix] = 1.
                mean[ix] = self.val_ar[w]['arousal']['mean']
                sd[ix] = self.val_ar[w]['arousal']['sd']

        total_sd = np.sum(sd) * tweet_arr
        with np.errstate(divide='ignore', invalid='ignore'):
            sd_ratio = total_sd / sd
            sd_ratio[sd == 0] = 0
        sd_weight = sd_ratio / np.sum(sd_ratio)

        if np.sum(mean*sd_weight) == np.nan:
            self.arousal_score = 0
        else:
            self.arousal_score = np.sum(mean*sd_weight)

        return self.arousal_score

    def get_sentiment_score(self):
        """
        Weights the posititive/negative sentiment of the tweet.
        """
        vectorized = self.nb_vectorizer.transform(self.stemmed)
        self.sentiment_score = np.average(1 - self.nb_model.predict_proba(vectorized)[:,1])

        return self.sentiment_score

    # def get_topics(self):
    #     """
    #     Extract the topics from the tweet using the LDA model.
    #     """
        # return self.model.get_document_topics(self.model.id2word.doc2bow(self.phrased), per_word_topics=False)

    def get_emoji_count(self):
        """
        Count the Mad! faces.
        """
        positives = ['\<f0\>\<U\+009F\>\<U\+0099\>\<U\+0082\>']
        outrage = ['\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A4\>', '\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A0\>', \
                '\<f0\>\<U\+009F\>\<U\+0098\>\<U\+00A1\>']
	positive_score = np.array([p in self.tweet for p in positives]).sum()
	outrage_score = np.array([o in self.tweet for o in outrage]).sum()
        self.emoji_count = outrage_score-positive_score
        return self.emoji_count

    def get_base_outrage_count(self):
        """
        Get the number of outrage words in the tweet.
        """
        self.base_outrage_count = 0
        for stem in self.stemmed:
            self.base_outrage_count += len(set([stem]) & set(self.outrage_list))
        return self.base_outrage_count

    def get_expanded_outrage_count(self):
        """
        Get the number of outrage words in the tweet.
        """
        self.expanded_outrage_count = 0
        for stem in self.stemmed:
            self.expanded_outrage_count += len(set([stem]) & set(self.exp_outrage_list))
        return self.expanded_outrage_count

    def get_outrage_score(self):
        """
        Uses the results of each of the index measures to create one score.
        0.16: (1 - df.scaled_topic_valence)
        0.16: df.scaled_topic_arousal
        0.15: (1 - df.scaled_valence)
        0.14: df.scaled_arousal
        0.14: df.scaled_outrage_words
        0.13: df.negative_sentiment_prob
        0.12: df.scaled_ext_outrage_words
        0.00: df.net_emo_outrage
        """
        # self.topics = self.get_topics()
        # topic_valence_score = 0
        # topic_arousal_score = 0
        # for tup in self.topics:
        #     topic_valence_score += self.topic_valence_scaler[tup[0]] * tup[1]
        #     topic_arousal_score += self.topic_arousal_scaler[tup[0]] * tup[1]

        scores = np.array([
            self.get_base_outrage_count(),
            self.get_expanded_outrage_count(),
            self.get_arousal_score(),
            (1 - self.get_valence_score()),
            self.get_sentiment_score(),
            self.get_emoji_count()
            # (1 - topic_valence_score),
            # topic_arousal_score
            ])
        # weights = np.array([0.14, 0.12, 0.14, 0.15, 0.13, 0.00, 0.16, 0.16])
        # self.outrage_meter = np.sum(scores*weights)

        # labels = ["outrage", "expanded outrage", "arousal", "valence", "sentiment", "emoji", "topic valence", "topic arousal", 'outrage score']
        labels = ["outrage", "expanded outrage", "arousal", "valence", "sentiment", "emoji", 'outrage score']
        # weights = np.array([0.2, 0.15, 0.15, 0.13, 0.11, 0.10, 0.08, 0.08])
        weights = np.array([0.2, 0.15, 0.15, 0.13, 0.11, 0.10])
        self.outrage_meter = np.sum(scores*weights)
        
        for i in range(len(scores)):
            print(labels[i] + ": " + str(scores[i]))
        
        scores = np.append(scores, self.outrage_meter)
        return scores

        # return self.outrage_meter
