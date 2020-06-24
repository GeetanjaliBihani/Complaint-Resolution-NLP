#Complaints data upload
import pandas as pd
from google.colab import files
uploaded = files.upload()

#Data setting
import io
text = pd.read_excel(io.BytesIO(uploaded["Complaints 2019 final.xlsx"]), index= False) 
text.columns=["Part", "Date", "Time", "Complaint", "Follow_up1", "Follow_up2", "blrb"]
text = text.drop(["blrb"], axis=1)
text['Complaint'] = text['Complaint'].astype(str)
text['Follow_up1'] = text['Follow_up1'].astype(str)
text['Follow_up2'] = text['Follow_up2'].astype(str)
text=text[1:]
text=text.reset_index(drop=True)

#Dividing complaint, followup1 and followup2 columns
data_text1 = text[['Complaint']]
data_text1['index'] = data_text1.index
documents1 = data_text1

data_text2 = text[['Follow_up1']]
data_text2['index'] = data_text2.index
documents2 = data_text2

data_text3 = text[['Follow_up2']]
data_text3['index'] = data_text3.index
documents3 = data_text3

### Convert to list, remove new line characters
#complaints
import gensim
import re
data1 = documents1["Complaint"].values.tolist()
data1 = [re.sub('\s+', ' ', sent) for sent in data1]
data1 = [str(sent) for sent in data1]
print(data1[:1])
#followup1
data2 = documents2["Follow_up1"].values.tolist()
data2 = [re.sub('\s+', ' ', sent) for sent in data2]
data2 = [str(sent) for sent in data2]
print(data2[:1])
#followup2
data3 = documents3["Follow_up2"].values.tolist()
data3 = [re.sub('\s+', ' ', sent) for sent in data3]
data3 = [str(sent) for sent in data3]
print(data3[:1])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words1 = list(sent_to_words(data1))
data_words2 = list(sent_to_words(data2))
data_words3 = list(sent_to_words(data3))


print(data_words1[:1])
print(data_words2[:1])
print(data_words3[:1])

# # Build the bigram models
bigram1 = gensim.models.Phrases(data_words1, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram2 = gensim.models.Phrases(data_words2, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram3= gensim.models.Phrases(data_words3, min_count=5, threshold=100) # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod1 = gensim.models.phrases.Phraser(bigram1)
bigram_mod2 = gensim.models.phrases.Phraser(bigram2)
bigram_mod3 = gensim.models.phrases.Phraser(bigram3)

# Define functions for stopwords, bigrams, trigrams and lemmatization
# NLTK Stop words
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams1(texts):
    return [bigram_mod1[doc] for doc in texts]
def make_bigrams2(texts):
    return [bigram_mod2[doc] for doc in texts]
def make_bigrams3(texts):
    return [bigram_mod3[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops1 = remove_stopwords(data_words1)
data_words_nostops2 = remove_stopwords(data_words2)
data_words_nostops3 = remove_stopwords(data_words3)

# # Form Bigrams
data_words_bigrams1 = make_bigrams1(data_words_nostops1)
data_words_bigrams2 = make_bigrams2(data_words_nostops2)
data_words_bigrams3 = make_bigrams3(data_words_nostops3)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
import spacy
# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized1 = lemmatization(data_words_bigrams1, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized2 = lemmatization(data_words_bigrams2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized3 = lemmatization(data_words_bigrams3, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


print(data_lemmatized1[:1])
print(data_lemmatized2[:1])
print(data_lemmatized3[:1])

#creating dictionaries
import gensim
dictionary1 = gensim.corpora.Dictionary(data_lemmatized1)
count = 0
print('dictonary1')
for k, v in dictionary1.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary2 = gensim.corpora.Dictionary(data_lemmatized2)
count = 0
print('dictonary2')
for k, v in dictionary2.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary3 = gensim.corpora.Dictionary(data_lemmatized3)
count = 0
print('dictonary3')
for k, v in dictionary3.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#create bow representation of words
bow_corpus1 = [dictionary1.doc2bow(doc) for doc in data_lemmatized1]
bow_corpus2 = [dictionary2.doc2bow(doc) for doc in data_lemmatized2]    
bow_corpus3 = [dictionary3.doc2bow(doc) for doc in data_lemmatized3]

#creating tfidf representations
from gensim import corpora, models
tfidf1 = models.TfidfModel(bow_corpus1)
corpus_tfidf1 = tfidf1[bow_corpus1]
from pprint import pprint
for doc in corpus_tfidf1:
    pprint(doc)
    break


from gensim import corpora, models
tfidf2 = models.TfidfModel(bow_corpus2)
corpus_tfidf2 = tfidf2[bow_corpus2]
from pprint import pprint
for doc in corpus_tfidf2:
    pprint(doc)
    break

from gensim import corpora, models
tfidf3 = models.TfidfModel(bow_corpus3)
corpus_tfidf3 = tfidf3[bow_corpus3]
from pprint import pprint
for doc in corpus_tfidf3:
    pprint(doc)
    break

from gensim.models import CoherenceModel
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
import gensim 
from gensim.models import CoherenceModel
model_list1, coherence_values1 = compute_coherence_values(dictionary=dictionary1, corpus=bow_corpus1, texts=data_lemmatized1, start=2, limit=20, step=2)
model_list2, coherence_values2 = compute_coherence_values(dictionary=dictionary2, corpus=bow_corpus2, texts=data_lemmatized2, start=2, limit=20, step=2)
model_list3, coherence_values3 = compute_coherence_values(dictionary=dictionary3, corpus=bow_corpus3, texts=data_lemmatized3, start=2, limit=20, step=2)

# Commented out IPython magic to ensure Python compatibility.
# Show graph complaint topics
import matplotlib.pyplot as plt
# %matplotlib inline
limit=20; start=2; step=2;
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
x = range(start, limit, step)
plt.plot(x, coherence_values1)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#running lda model 
import gensim
lda_model1 = gensim.models.LdaMulticore(bow_corpus1, num_topics=5, id2word=dictionary1, passes=2, workers=2)
lda_model2 = gensim.models.LdaMulticore(bow_corpus2, num_topics=5, id2word=dictionary2, passes=2, workers=2)
lda_model3 = gensim.models.LdaMulticore(bow_corpus3, num_topics=5, id2word=dictionary3, passes=2, workers=2)

#explore each topic
for idx, topic in lda_model3.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

##Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model3.show_topics(formatted=False)

fig, axes = plt.subplots(1,5, figsize=(20,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords1 = format_topics_sentences(ldamodel=lda_model1, corpus=bow_corpus1, texts=data1)
df_topic_sents_keywords2 = format_topics_sentences(ldamodel=lda_model2, corpus=bow_corpus2, texts=data2)
df_topic_sents_keywords3 = format_topics_sentences(ldamodel=lda_model3, corpus=bow_corpus3, texts=data3)

# Format
df_dominant_topic1 = df_topic_sents_keywords1.reset_index()
df_dominant_topic1.columns = ['Document_No', 'Topic1', 'Topic_contri1', 'Keywords1', 'Complaint']
# Format
df_dominant_topic2 = df_topic_sents_keywords2.reset_index()
df_dominant_topic2.columns = ['Document_No', 'Topic2', 'Topic_contri2', 'Keywords2', 'Follow_up1']
# Format
df_dominant_topic3 = df_topic_sents_keywords3.reset_index()
df_dominant_topic3.columns = ['Document_No', 'Topic3', 'Topic_contri3', 'Keywords3', 'Follow_up2']
# Show
df_dominant_topic1.head(10)

df_dominant_topic1.tail(10)

# Show
df_dominant_topic1.head(5)

# Show
df_dominant_topic2.head(5)

# Show
df_dominant_topic3.head(5)

##merge all dominant topic information in one table
text['Document_No']=text.reset_index().index
df_merge_col = pd.merge(text, df_dominant_topic1[['Document_No','Topic1', 'Topic_contri1', 'Keywords1']], on='Document_No', how='left')
df_merge_col1 = pd.merge(df_merge_col, df_dominant_topic2[['Document_No','Topic2', 'Topic_contri2', 'Keywords2']], on='Document_No', how='left')
text = pd.merge(df_merge_col1, df_dominant_topic3[['Document_No','Topic3', 'Topic_contri3', 'Keywords3']], on='Document_No', how='left')

#topic distribution visualization
#for complaints topic model
!pip install pyLDAvis
import pyLDAvis
from pyLDAvis import gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model1, bow_corpus1, dictionary1)
vis

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
rep1=vectorizer.fit_transform(data1+data2+data3)
pairwise_similarity = rep1 * rep1.T 
pairwise_similarity=pairwise_similarity.toarray()
pairwise_similarities=pd.DataFrame(pairwise_similarity)

a=[]
for i in range(0,1317):
  s= pairwise_similarity[i][i+1317]
  a.append(s)
a=pd.DataFrame(a)
a.columns=["cos_sim1"]

b=[]
for i in range(0,1317):
  t= pairwise_similarity[i][i+2634]
  b.append(t)
b=pd.DataFrame(b)
b.columns=["cos_sim2"]

c=[]
for i in range(1317,2634):
  u= pairwise_similarity[i][i+1317]
  c.append(u)
c=pd.DataFrame(c)
c.columns=["cos_sim3"]

text_new=pd.concat([text,a,b,c], axis=1)
##final dataset with complaint topics, followup1 topics, followup2 topics and respective cosine similarity features
text_new1=text_new[["Part", "Topic1","Topic_contri1","Topic2","Topic_contri2","Topic3","Topic_contri3","cos_sim1", "cos_sim2", "cos_sim3"]]

text_new1.head()

import numpy as np
#finding top 3 recommendations for followup 1
##training a classifier to find followup topic for each problem topic
##data for training followup1 model
df1 = text_new1.iloc[:922]
df2 = text_new1.iloc[922:]
df1=df1[["Part","Topic1", "Topic_contri1","Topic_contri2","Topic3","Topic_contri3","cos_sim1", "cos_sim2", "cos_sim3", "Topic2"]]
df2=df2[["Part","Topic1", "Topic_contri1","Topic_contri2","Topic3","Topic_contri3","cos_sim1", "cos_sim2", "cos_sim3", "Topic2"]]

y_train = df1.iloc[:,9]
X_train = df1.iloc[:,1:9]

y_test = df2.iloc[:,9]
X_test = df2.iloc[:,1:9]

##training an SVM classifier
import pandas as pd
import os

from sklearn.svm import SVC 
SVM = SVC(kernel = 'linear', probability=True).fit(X_train, y_train) 

svc_pred_train=SVM.predict(X_train)

#finding train and test accuracies
from sklearn.metrics import confusion_matrix 
svc_pred_test = SVM.predict(X_test) 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('training accuracy:',accuracy_score(y_train, svc_pred_train))
print('testing accuracy:',accuracy_score(y_test, svc_pred_test))

##finding train and test probabilities
rank_prob_test=SVM.predict_proba(X_test)
rank_prob_train=SVM.predict_proba(X_train)

##convert probability information into df
rank_prob_train = pd.DataFrame(data=rank_prob_train[0:,0:])
rank_prob_test = pd.DataFrame(data=rank_prob_test[0:,0:])
rank_prob_test.index = np.arange(922, len(rank_prob_test) + 922)

##merge proba info with test and train data
df=pd.concat([df1,df2])
rank_df=pd.concat([rank_prob_train,rank_prob_test])
rank_df.columns=["f1t0", "f1t1", "f1t2", "f1t3", "f1t4"]
df_rank=pd.concat([df,rank_df], axis=1)

##finding top 3 recommendations using classifier probabilites

df_rank["max1"]=df_rank.iloc[:,10:15].max(axis=1)
df_rank["max2"]=df_rank.iloc[:,10:15].apply(lambda row: row.nlargest(2).values[-1],axis=1)
df_rank["max3"]=df_rank.iloc[:,10:15].apply(lambda row: row.nlargest(3).values[-1],axis=1)

df_rank["f1_reco1"]=1
df_rank["f1_reco2"]=1
df_rank["f1_reco3"]=1
for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,15]:
      if i == 10:
        df_rank["f1_reco1"].iloc[j]="0.0"
      if i == 11:
        df_rank["f1_reco1"].iloc[j]="1.0"
      if i == 12:
        df_rank["f1_reco1"].iloc[j]="2.0"
      if i == 13:
        df_rank["f1_reco1"].iloc[j]="3.0"
      if i == 14:
        df_rank["f1_reco1"].iloc[j]="4.0"

for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,16]:
      if i == 10:
        df_rank["f1_reco2"].iloc[j]="0.0"
      if i == 11:
        df_rank["f1_reco2"].iloc[j]="1.0"
      if i == 12:
        df_rank["f1_reco2"].iloc[j]="2.0"
      if i == 13:
        df_rank["f1_reco2"].iloc[j]="3.0"
      if i == 14:
        df_rank["f1_reco2"].iloc[j]="4.0"

for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,17]:
      if i == 10:
        df_rank["f1_reco3"].iloc[j]="0.0"
      if i == 11:
        df_rank["f1_reco3"].iloc[j]="1.0"
      if i == 12:
        df_rank["f1_reco3"].iloc[j]="2.0"
      if i == 13:
        df_rank["f1_reco3"].iloc[j]="3.0"
      if i == 14:
        df_rank["f1_reco3"].iloc[j]="4.0"

df_rankf1=df_rank
df_rankf1.head()

#finding top 3 recommendations for followup 2
##training a classifier to find followup topic for each problem topic
##data for training followup1 model
df1 = text_new1.iloc[:922]
df2 = text_new1.iloc[922:]
df1=df1[["Part","Topic1", "Topic_contri1","Topic2","Topic_contri2","Topic_contri3","cos_sim1", "cos_sim2", "cos_sim3", "Topic3" ]]
df2=df2[["Part","Topic1", "Topic_contri1","Topic2","Topic_contri2","Topic_contri3","cos_sim1", "cos_sim2", "cos_sim3", "Topic3"]]

y_train = df1.iloc[:,9]
X_train = df1.iloc[:,1:9]

y_test = df2.iloc[:,9]
X_test = df2.iloc[:,1:9]

##training an SVM classifier
import pandas as pd
import os

from sklearn.svm import SVC 
SVM = SVC(kernel = 'linear', probability=True).fit(X_train, y_train) 

svc_pred_train=SVM.predict(X_train)

#finding train and test accuracies
from sklearn.metrics import confusion_matrix 
svc_pred_test = SVM.predict(X_test) 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('training accuracy:',accuracy_score(y_train, svc_pred_train))
print('testing accuracy:',accuracy_score(y_test, svc_pred_test))

##finding train and test probabilities
rank_prob_test=SVM.predict_proba(X_test)
rank_prob_train=SVM.predict_proba(X_train)

##convert probability information into df
rank_prob_train = pd.DataFrame(data=rank_prob_train[0:,0:])
rank_prob_test = pd.DataFrame(data=rank_prob_test[0:,0:])
rank_prob_test.index = np.arange(922, len(rank_prob_test) + 922)

##merge proba info with test and train data
df=pd.concat([df1,df2])
rank_df=pd.concat([rank_prob_train,rank_prob_test])
rank_df.columns=["f2t0", "f2t1", "f2t2", "f2t3", "f2t4"]
df_rank=pd.concat([df,rank_df], axis=1)

##finding top 3 recommendations using classifier probabilites

df_rank["max1_f2"]=df_rank.iloc[:,10:15].max(axis=1)
df_rank["max2_f2"]=df_rank.iloc[:,10:15].apply(lambda row: row.nlargest(2).values[-1],axis=1)
df_rank["max3_f2"]=df_rank.iloc[:,10:15].apply(lambda row: row.nlargest(3).values[-1],axis=1)

df_rank["f2_reco1"]=1
df_rank["f2_reco2"]=1
df_rank["f2_reco3"]=1
for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,15]:
      if i == 10:
        df_rank["f2_reco1"].iloc[j]="0.0"
      if i == 11:
        df_rank["f2_reco1"].iloc[j]="1.0"
      if i == 12:
        df_rank["f2_reco1"].iloc[j]="2.0"
      if i == 13:
        df_rank["f2_reco1"].iloc[j]="3.0"
      if i == 14:
        df_rank["f2_reco1"].iloc[j]="4.0"

for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,16]:
      if i == 10:
        df_rank["f2_reco2"].iloc[j]="0.0"
      if i == 11:
        df_rank["f2_reco2"].iloc[j]="1.0"
      if i == 12:
        df_rank["f2_reco2"].iloc[j]="2.0"
      if i == 13:
        df_rank["f2_reco2"].iloc[j]="3.0"
      if i == 14:
        df_rank["f2_reco2"].iloc[j]="4.0"

for j in range(0,1317):
  for i in range(10,15):
    if df_rank.iloc[j, i] == df_rank.iloc[j,17]:
      if i == 10:
        df_rank["f2_reco3"].iloc[j]="0.0"
      if i == 11:
        df_rank["f2_reco3"].iloc[j]="1.0"
      if i == 12:
        df_rank["f2_reco3"].iloc[j]="2.0"
      if i == 13:
        df_rank["f2_reco3"].iloc[j]="3.0"
      if i == 14:
        df_rank["f2_reco3"].iloc[j]="4.0"
df_rankf2=df_rank
df_rankf2.head()

df_rankf1_n=df_rankf1[["f1_reco1","f1_reco2","f1_reco3"]]
df_rankf1_final=pd.concat([text,df_rankf1_n], axis=1)
df_rankf2_n=df_rankf2[["f2_reco1","f2_reco2","f2_reco3"]]
df_rankf2_final=pd.concat([text,df_rankf2_n], axis=1)

df_rankf1_final.head(5)

df_rankf2_final.head(5)

from google.colab import drive
drive.mount('drive')

df_rankf1_final.to_csv('followup1_recos_new1.csv')
!cp followup1_recos_new1.csv drive/My\ Drive/
df_rankf2_final.to_csv('followup2_recos_new2.csv')
!cp followup2_recos_new2.csv drive/My\ Drive/

