from __future__ import print_function
print('==================== Importing Packages ====================')

import warnings
warnings.filterwarnings("ignore")

import gensim

#For Feature Engineering
import re
import random
from scipy import sparse as sp
import string
import nltk
from nltk.tokenize import wordpunct_tokenize,TweetTokenizer
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import tensorflow as tf
import random as rn
import os
from tensorflow.keras import optimizers,backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input,Dense, Dropout,Activation,BatchNormalization
from tensorflow.keras import regularizers

#------------------------------------------------------------------------------------------
print("==================== Importing Supporting Files ====================")

#Sentiment Training Data
df_train_sentiment = pd.read_csv('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/df_train_final.csv')
df_train_sentiment=df_train_sentiment.drop(columns=['Unnamed: 0','index','W2V_50_Means','GloVe40_Means','NER_DATE','NER_DURATION','NER_LOCATION','NER_MONEY''NER_NUMBER','NER_ORDINAL','NER_ORGANIZATION','NER_PERCENT','NER_PERSON','NER_SET','NER_TIME','created_at','official_account','sentiment','total_likes'])

#Google Word2Vec Pretrained Model 
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/GoogleNews-vectors-negative300.bin', binary=True) 

#------------------------------------------------------------------------------------------
print("==================== Importing Data ====================")

df_stocktwits = pd.read_csv('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Dataset/Final/df_stocktwits_prepared_final.csv')
df_yfinance = pd.read_csv('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Dataset/Final/df_yfinance_BA_prepared.csv')

#------------------------------------------------------------------------------------------
print("==================== Importing Trained Model ====================")

model_MLP = load_model('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/model_MLP.h5')  
model_MLP_W2V_Sentence_Vector = load_model('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/model_MLP_W2V_Sentence_Vector.h5')
model_CNN_W2V = load_model('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/model_CNN_W2V.h5')
model_LSTM_W2V = load_model('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/model_LSTM_W2V.h5')
model_sentiment_ensemble = load_model('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Supporting_Files/model_Ensemble.h5')

#------------------------------------------------------------------------------------------

def model_sentiment_pred(df_test):
  '''
  Function to predict sentiment score
  '''

  pred_MLP=MLP_model_pred(model_MLP,df_train_sentiment,df_test,sentence_vector=False)
  print('================= Done MLP Feature Driven =================')
  pred_MLP_W2V_Sentence_Vector=MLP_model_pred(model_MLP_W2V_Sentence_Vector,df_train_sentiment,df_test,sentence_vector=True)
  print('================= Done MLP Sentence Vector =================')
  pred_CNN=model_CNN_LSTM_pred(model_CNN_W2V,df_train_sentiment,df_test)
  print('================= Done CNN =================')
  pred_LSTM=model_CNN_LSTM_pred(model_LSTM_W2V,df_train_sentiment,df_test)
  print('================= Done LSTM =================')

  X_test=pd.DataFrame(pred_MLP,columns=['pred_MLP_Feature_Driven'])
  X_test['pred_MLP_W2V_Sentence_Vector']=pred_MLP_W2V_Sentence_Vector
  X_test['pred_CNN_W2V']=pred_CNN
  X_test['pred_lstm_W2V']=pred_LSTM
  print('================= Done Preparing Data =================')

  #Predict 
  y_pred=model_sentiment_ensemble.predict(X_test,batch_size=16)
  y_pred=pd.Series(y_pred.tolist()).apply(lambda x: x[0])
  return y_pred


def MLP_model_pred(model,df_train,df_test,sentence_vector=False):
  '''
  Function to predict sentiment score using MLP
  '''

  #Filter Data
  if not sentence_vector:
    #Feature Engineering
    df_train,df_test=PMI(df_train,df_test)
    for gram in [1,2,3,4]:
      df_train,df_test=rf_ngram(df_train,df_test,gram=gram)

    df_train=df_train.drop(['cashtag','spans','text','clean_text','base_text','source'],1)
    df_test=df_test.drop(['clean_text','base_text'],1)
  else:
    df_train=W2V_sentence_embedding(df_train)
    df_test=W2V_sentence_embedding(df_test)

  #Split data into dependent and independent variable
  if 'sentiment score' in df_train.columns.tolist():
    X_train=df_train.drop(['sentiment score'],1)
  else:
    X_train=df_train.copy()
  X_test=df_test.copy()

  #Impute Missing Testues
  imputer = KNNImputer(n_neighbors=3)
  X_train=pd.DataFrame(imputer.fit_transform(X_train))
  X_test_split = np.array_split(X_test, 20)
  X_test_pool=pd.DataFrame(imputer.fit_transform(X_test_split[0]))
  for i in range(1,20):
    X_imputed=pd.DataFrame(imputer.fit_transform(X_test_split[i]))
    X_test_pool=pd.concat([X_test_pool,X_imputed],ignore_index=True)
  X_test=X_test_pool.copy()

  #Predict 
  y_pred=model.predict(X_test,batch_size=32)
  y_pred=pd.Series(y_pred.tolist()).apply(lambda x: x[0])
  return y_pred


def model_CNN_LSTM_pred(model,df_train,df_test):
  '''
  Function to predict sentiment score using CNN / LSTM
  '''

  X_train=df_train['clean_text'].tolist()
  X_test=df_test['clean_text'].tolist()

  # prepare tokenizer
  t = Tokenizer()
  t.fit_on_texts(X_train)
  X_train = t.texts_to_sequences(X_train)
  X_test = t.texts_to_sequences(X_test)

  # Adding 1 because of reserved 0 index
  vocab_size = len(t.word_index) + 1

  X_train = pad_sequences(X_train, padding='post', maxlen=50)
  X_test = pad_sequences(X_test, padding='post', maxlen=50)

  #Predict 
  y_pred=model.predict(X_test,batch_size=32)
  y_pred=pd.Series(y_pred.tolist()).apply(lambda x: x[0])
  return y_pred


def tokenize(sentence):
  '''
  tokenize input sentence into token
  '''
  return (nltk.regexp_tokenize(sentence, pattern=r"\s|[\.,;]\D", gaps=True))


def n_grams_handled(sentence):
  '''
  Filter before generate n-gram
  '''
  try:
    tk=TweetTokenizer()
    cashtag_pat=r'\$[^\s]+'
    hashtag_pat=r'#([^\s]+)'
    word_number_pat=r'\w*\d\w*'
  
    #Remove word which has length < 2
    stripped=' '.join([word for word in sentence.split() if len(word)>=2])
  
    #Remove hashtag
    hashtag_handled= re.sub(hashtag_pat,"", stripped)
  
    #Remove cashtag
    cashtag_handled= re.sub(cashtag_pat,"", hashtag_handled)
    
    #Remove word with number
    number_handled= re.sub(word_number_pat,"", cashtag_handled)
  
    #Remove unnecesary white spaces
    words = tk.tokenize(number_handled)
    words = [x for x in words if x not in string.punctuation]
    clean_sentence=(" ".join(words)).strip()
    return  clean_sentence
  except:
    return sentence


def rf_ngram(df_train,df_test,gram): 
  '''
  create rf-ngram
  '''
  def sentence_sparse(sentence,gram,rf_ngram,sparse_rf_ngram):
    #Initiate Linke List Sparse Matrix
    zero_sparse=sp.lil_matrix( (1,len(rf_ngram)), dtype=float)
    #Assign Value of rf_ngram to each word in sentence
    splitted_text=tokenize(n_grams_handled(sentence))
    #Unigram
    if gram==1:
      for word in splitted_text:
        if word in rf_ngram.index:
          zero_sparse[0,rf_ngram.index.get_loc(word)]+=sparse_rf_ngram[0,rf_ngram.index.get_loc(word)]
      #Convert LinkedList Sparse Matrix into CSR Sparse Matrix
      sparse=zero_sparse.tocsr()
    #Bigram
    elif gram==2:
      bigram=lambda x: splitted_text[x]+' '+splitted_text[x+1]
      it_2_gram=range(len(splitted_text)-1)
      for i in it_2_gram:
        if bigram(i) in rf_ngram.index:
          zero_sparse[0,rf_ngram.index.get_loc(bigram(i))]+=sparse_rf_ngram[0,rf_ngram.index.get_loc(bigram(i))]
      #Convert LinkedList Sparse Matrix into CSR Sparse Matrix
      sparse=zero_sparse.tocsr()
    #Trigram
    elif gram==3:
      trigram=lambda x: splitted_text[x]+' '+splitted_text[x+1]+' '+splitted_text[x+2]
      it_3_gram=range(len(splitted_text)-2)
      for i in it_3_gram:
        if trigram(i) in rf_ngram.index:
          zero_sparse[0,rf_ngram.index.get_loc(trigram(i))]+=sparse_rf_ngram[0,rf_ngram.index.get_loc(trigram(i))]
      #Convert LinkedList Sparse Matrix into CSR Sparse Matrix
      sparse=zero_sparse.tocsr()
    #4grams
    elif gram==4:
      fourgram=lambda x: splitted_text[x]+' '+splitted_text[x+1]+' '+splitted_text[x+2]+' '+splitted_text[x+3]
      it_4_gram=range(len(splitted_text)-3)
      for i in it_4_gram:
        if fourgram(i) in rf_ngram.index:
          zero_sparse[0,rf_ngram.index.get_loc(fourgram(i))]+=sparse_rf_ngram[0,rf_ngram.index.get_loc(fourgram(i))]
      #Convert LinkedList Sparse Matrix into CSR Sparse Matrix
      sparse=zero_sparse.tocsr()
    return(sparse)

  BOW_df= pd.DataFrame(columns=['pos','neutral','neg'])
  words_set = set()
  
  #Creating the rf_ngram dictionary of words
  it=range(len(df_train))
  for i in it:
    score=df_train.loc[i,'sentiment score']
    if score>0:
      score='pos'
    elif score<0:
      score='neg'
    else:
      score='neutral'
    try:
      text=df_train.loc[i,'clean_text']
      cleaned_text=n_grams_handled(text)
      splitted_text=tokenize(cleaned_text)
      if gram==1:
        for word in splitted_text:
          if word not in words_set:#check if this word already counted or not in the full corpus
            words_set.add(word)
            BOW_df.loc[word] = [0,0,0]
            BOW_df.loc[word,score]+=1
          else:
            BOW_df.loc[word,score]+=1
      elif gram==2:
        it_2_gram=range(len(splitted_text)-1)
        bigram=lambda x: splitted_text[x]+' '+splitted_text[x+1]
        for i in it_2_gram:
          if bigram(i) not in words_set:
            words_set.add(bigram(i))
            BOW_df.loc[bigram(i)] = [0,0,0]
            BOW_df.loc[bigram(i),score]+=1
          else:
            BOW_df.loc[bigram(i),score]+=1
      elif gram==3:
        it_3_gram=range(len(splitted_text)-2)
        trigram=lambda x: splitted_text[x]+' '+splitted_text[x+1]+' '+splitted_text[x+2]
        for i in it_3_gram:
          if trigram(i) not in words_set:
            words_set.add(trigram(i))
            BOW_df.loc[trigram(i)] = [0,0,0]
            BOW_df.loc[trigram(i),score]+=1
          else:
            BOW_df.loc[trigram(i),score]+=1
      elif gram==4:
        it_4_gram=range(len(splitted_text)-3)
        fourgram=lambda x: splitted_text[x]+' '+splitted_text[x+1]+' '+splitted_text[x+2]+' '+splitted_text[x+3]
        for i in it_4_gram:
          if fourgram(i) not in words_set:
            words_set.add(fourgram(i))
            BOW_df.loc[fourgram(i)] = [0,0,0]
            BOW_df.loc[fourgram(i),score]+=1
          else:
            BOW_df.loc[fourgram(i),score]+=1 
    except:
      None
  #Calculate rf_ngram for each word
  series_1=pd.Series([1 for x in range(len(BOW_df))])
  series_1.index=BOW_df.index
  series_2=pd.Series([2 for x in range(len(BOW_df))])
  series_2.index=BOW_df.index
  frac_1=np.log(series_2+(BOW_df['pos']/pd.concat([series_1,BOW_df['neg']],1).max(axis=1)))
  frac_2=np.log(series_2+(BOW_df['neg']/pd.concat([series_1,BOW_df['pos']],1).max(axis=1)))
  rf_ngram_series= pd.concat([frac_1,frac_2],1).max(axis=1)
  sparse_rf_ngram=sp.csr_matrix(rf_ngram_series)

  def rf_ngram_calculate(x):
    lst=[i for i in sentence_sparse(x,gram,rf_ngram_series,sparse_rf_ngram).toarray()[0].tolist() if i!=0]
    if type(x)!=str:
      return(np.nan)
    else:
      if len(lst)>0:
        return(np.mean(lst))
      else:
        return(np.nan)

  rf_ngram_avg_list_train=df_train['clean_text'].apply(lambda x: rf_ngram_calculate(x))
  rf_ngram_avg_list_test=df_test['clean_text'].apply(lambda x: rf_ngram_calculate(x))

  df_train['Avg_rf_'+str(gram)+'-grams']= rf_ngram_avg_list_train
  df_test['Avg_rf_'+str(gram)+'-grams']= rf_ngram_avg_list_test

  return(df_train,df_test)


def PMI(df_train,df_test): 
  '''
  create PMI variable
  '''
  BOW_df= pd.DataFrame(columns=['pos','neutral','neg'])
  words_set = set()
  
  #Creating the dictionary of words
  it=range(len(df_train))
  for i in it:
    score=df_train.loc[i,'sentiment score']
    if score>0:
      score='pos'
    elif score<0:
      score='neg'
    else:
      score='neutral'
    try:
      text=df_train.loc[i,'clean_text']
      cleaned_text=n_grams_handled(text)
      splitted_text=tokenize(cleaned_text)
      for word in splitted_text:
        if word not in words_set:#check if this word already counted or not in the full corpus
          words_set.add(word)
          BOW_df.loc[word] = [0,0,0]
          BOW_df.loc[word,score]+=1
        else:
          BOW_df.loc[word,score]+=1
    except:
      None
    
  N=len(BOW_df) #Number of unique tokens in the corpus
  pos_N=len(BOW_df[BOW_df.pos!=0]) #Number of unique positive tokens in the corpus
  neg_N=len(BOW_df[BOW_df.neg!=0]) #Number of unique positive tokens in the corpus
  total=BOW_df.sum().sum() #Number of tokens in the corpus
  pos_total=BOW_df.sum()['pos'] #Number of tokens in the positive corpus
  neg_total=BOW_df.sum()['neg'] #Number of tokens in the negative corpus
  PMI_df=pd.DataFrame(columns=['freq_word','freq_word_pos','freq_word_neg'])
  PMI_df['freq_word']=pd.Series(BOW_df.index).apply(lambda x: (BOW_df.loc[x,'pos']+BOW_df.loc[x,'neutral']+BOW_df.loc[x,'neg'])/total)
  PMI_df['freq_word_pos']=pd.Series(BOW_df.index).apply(lambda x: BOW_df.loc[x,'pos']/pos_total) #Freq of word w in positive text
  PMI_df['freq_word_neg']=pd.Series(BOW_df.index).apply(lambda x: BOW_df.loc[x,'neg']/neg_total) #Freq of word w in negative text
  PMI_df.index=BOW_df.index
  
  #Calculate PMI for each word
  PMI_df['PMI_pos']=np.log2(1+((PMI_df['freq_word_pos']*N)/(PMI_df['freq_word']*pos_N)))
  PMI_df['PMI_neg']=np.log2(1+((PMI_df['freq_word_neg']*N)/(PMI_df['freq_word']*neg_N)))
  PMI_df['PMI']=PMI_df['PMI_pos']-PMI_df['PMI_neg']
 
  def PMI_calculate(x):
    lst=[PMI_df.loc[i,'PMI'] for i in tokenize(n_grams_handled(x)) if i in PMI_df.index]
    if type(x)!=str:
      return(np.nan)
    else:
      if len(lst)>0:
        return(np.mean(lst))
      else:
        return(np.nan)

  PMI_avg_list_train=df_train['clean_text'].apply(lambda x: PMI_calculate(x))
  PMI_avg_list_test=df_test['clean_text'].apply(lambda x: PMI_calculate(x))
  
  df_train['PMI_score']=PMI_avg_list_train
  df_test['PMI_score']=PMI_avg_list_test

  return(df_train,df_test)


def W2V_sentence_embedding(df,span=False):
  '''
  return dataframe for W2V sentence embedding
  '''
  if not span:
    column='clean_text'
  else:
    column='spans'
    
  zero=np.array([float(0) for i in range(300)])
  nan=np.array([np.nan for i in range(300)])
  vec_W2V=lambda x: model_w2v[x] if x in model_w2v else zero
  W2V_avg=df[column].apply(lambda sent: pd.Series(tokenize(sent)).apply(lambda x: vec_W2V(x)).mean() if type(sent)==str else nan)
  
  W2V_df=pd.DataFrame(dict(zip(W2V_avg.index, W2V_avg.values))).T
  if 'sentiment score' in df.columns.tolist():
    W2V_df['sentiment score']=df['sentiment score']
    col=['x'+str(i) for i in W2V_df.drop(['sentiment score'],1).columns.tolist()]
    col.append('sentiment score')
  else:
    col=['x'+str(i) for i in W2V_df.columns.tolist()]
  W2V_df.columns=col
  return(W2V_df)


def main():
  #Predict Sentiment Score
  df_stocktwits['sentiment score']=model_sentiment_pred(df_stocktwits.drop(columns=['time','created_date']))

  #Create primary key column
  df_stocktwits['join']=df_stocktwits.apply(lambda x: str(pd.to_datetime(x['created_date']).date())+' '+str(x.time) ,axis=1)
  df_yfinance['join']=df_yfinance.apply(lambda x: str(pd.to_datetime(x['Unnamed: 0']).date())+' '+str(x.time) ,axis=1)

  #Aggregate Sentiment Score
  mean_sentiment_score_dict=df_stocktwits[['join','sentiment score']].groupby(['join']).mean().to_dict()['sentiment score']
  median_sentiment_score_dict=df_stocktwits[['join','sentiment score']].groupby(['join']).median().to_dict()['sentiment score']
  std_sentiment_score_dict=df_stocktwits[['join','sentiment score']].groupby(['join']).std().to_dict()['sentiment score']
  max_sentiment_score_dict=df_stocktwits[['join','sentiment score']].groupby(['join']).max().to_dict()['sentiment score']
  min_sentiment_score_dict=df_stocktwits[['join','sentiment score']].groupby(['join']).min().to_dict()['sentiment score']

  #Join the Aggregated Sentiment Score to Historical Stock Data
  df_yfinance['mean_sentiment_score']=df_yfinance['join'].apply(lambda x: mean_sentiment_score_dict[x] if x in mean_sentiment_score_dict else np.nan)
  df_yfinance['median_sentiment_score']=df_yfinance['join'].apply(lambda x: median_sentiment_score_dict[x] if x in median_sentiment_score_dict else np.nan)
  df_yfinance['std_sentiment_score']=df_yfinance['join'].apply(lambda x: std_sentiment_score_dict[x] if x in std_sentiment_score_dict else np.nan)
  df_yfinance['max_sentiment_score']=df_yfinance['join'].apply(lambda x: max_sentiment_score_dict[x] if x in max_sentiment_score_dict else np.nan)
  df_yfinance['min_sentiment_score']=df_yfinance['join'].apply(lambda x: min_sentiment_score_dict[x] if x in min_sentiment_score_dict else np.nan)

  #Drop primary key column
  df_yfinance=df_yfinance.drop(columns=['join'])

  #Export Resulting Data
  df_yfinance.to_csv('C:/Users/Louis Owen/Desktop/ICoDSA 2020/SENN/Dataset/Final/df_final.csv',index=False)


if __name__ == '__main__':
  main()