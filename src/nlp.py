# Import pkgs
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#pip install -U spacy download en_core_web_sm
import spacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from spacy.lang.en.examples import sentences



# Importing Gensim for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# PLotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# read in the text file from appropriate directory
df = pandas.read_csv('/path/to/file.csv')
df.count(), df.columns

# extract the text from html in body column and store in reviewtext column
df['feedback_f'] = df['feedback'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

# tokenize the text and store as new column, can clean this further later
df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['feedback_f']), axis=1)

# initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# perform sentiment analysis using Vader pkg and store as series variable
sentiment = df['feedback_f'].apply(lambda x: analyzer.polarity_scores(x))

# concatenate the sentiment variable with your existing df
df = pd.concat([df,sentiment.apply(pd.Series)],1)


###### EXTRACTING KEY WORDS

def pre_process(text):

    # lowercase
    text=text.lower()

    #remove tags
    text=re.sub("<!--?.*?-->","",text)

    # remove special characters and digits
    text=re.sub("(\d|\W)+"," ",text)

    return text

# apply pre-processing fn to the text
df['cleantext'] = df['feedback_f'].apply(lambda x:pre_process(x))

# get the text column change to list
docs=df['cleantext'].tolist()

#create a vocabulary of words, rule is to ignore words that appear in 80% of documents,
#eliminate stop words, set max features to 10k
cv=CountVectorizer(max_df=0.80,max_features=10000)
word_count_vector=cv.fit_transform(docs)
list(cv.vocabulary_.keys())[:10]


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# you only needs to do this once, this is a mapping of index to
feature_names=cv.get_feature_names()


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results


def get_keywords(idx):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)

    return keywords


# now create empty list to hold key words
kw = []

# run keywords fn and append to list
for idx in range(len(df)):
    kw.append(get_keywords(idx))

# add the list to the df as a new column
df['keywordsnew'] = kw


stop_words = set(stopwords.words('english'))

# Convert to list
data = df.feedback_f.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

print(data[:1])


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

print(data_words[:1])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load('en_core_web_sm')


data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


df['text_lemmatized'] = data_lemmatized


id2word = corpora.Dictionary(df['text_lemmatized'])

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id2word,
                                          num_topics=10,
                                          update_every=1,
                                          chunksize=100,
                                          passes=10,
                                          alpha='auto',
                                          per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

pyLDAvis.display(vis)

print('\nPerplexity: ', lda_model.log_perplexity(corpus))

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

pyLDAvis.display(vis)
