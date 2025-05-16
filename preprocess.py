import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def clean(df, column):
    # preprocessing

    df[column] = df[column].str.lower()

    STOPWORDS = set(stopwords.words('english'))

    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in STOPWORDS])

    df[column] = df[column].apply(lambda x: remove_stopwords(x))

    def remove_splcharacters(text):
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text

    df[column] = df[column].apply(lambda x: remove_splcharacters(x))

    return df

def lemmatize(df, column):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        return " ".join([lemmatizer.lemmatize(word, pos ='v') for word in text.split()])

    df[column] = df[column].apply(lambda x: lemmatize_text(x))

    return df

def tokenize_pad(df, column):
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df[column])
    word_index = tokenizer.word_index

    # padding the data
    sequence = tokenizer.texts_to_sequences(df[column])

    # maximum length of the data
    max_len = max(len(data) for data in sequence)
    padded_seq = pad_sequences(sequence,maxlen=max_len, padding='post', truncating= 'post')

    return padded_seq, word_index

def word2vec(word_index):

    #create embedding index
    embedding_index = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embed_list = np.array(values[1:]).astype('float32')
            embedding_index[word]= embed_list
    
    #create embedding matrix
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size+1, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def whatsappchat_clean(df):
    df=df.drop(0)
    df=df.drop(columns=[0])
    df.columns=['Chat']
    Message=df["Chat"].str.split("-",n=1,expand=True)
    Message1=Message[1].str.split(":",n=1,expand=True)
    df["Name"]=Message1[0]
    df["Chat"]=Message1[1]
    df=df[["Name","Chat"]]
    return df

def call_all(df, column):
    df = clean(df, column)
    df = lemmatize(df, column)

   # padded_seq, word_index = tokenize_pad(df, column)

    return df