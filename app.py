
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import preprocess
import plot_function  # your plotting functions (make sure they return matplotlib figs or plotly figs)
import base64
import re
from datetime import datetime


# Set background image
def set_bg_from_local(img_path):
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
    encoded = base64.b64encode(img_data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def parse_chat(uploaded_file):
    # Read the uploaded file content and decode it
    raw_text = uploaded_file.read().decode('utf-8')

    # Pattern to match lines: date, time, name, message
    pattern = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?(?:am|pm)) - (.*?): (.*)'

    messages = []
    for line in raw_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            date_str, time_str, name, message = match.groups()

            # ✅ Skip unwanted content
            if "<Media omitted>" in message or "(file attached)" in message:
                continue

            # Fix special space character in time if present
            time_str = time_str.replace('\u202f', ' ')
            dt = datetime.strptime(date_str + " " + time_str, "%d/%m/%y %I:%M %p")

            messages.append([name, dt, message])
        elif messages:
            # Continuation of previous message
            if "<Media omitted>" in line or "(file attached)" in line:
                continue
            messages[-1][2] += "\n" + line

    df = pd.DataFrame(messages, columns=['Name', 'Date', 'Chat'])

    # Remove empty or whitespace-only chats
    df = df.dropna(subset=['Chat'])
    df = df[df['Chat'].str.strip() != '']

    return df[['Name', 'Chat']]


set_bg_from_local("Images/bg.png")

st.title("Whatsapp Chat Emotion Detection")

uploaded_file = st.file_uploader("Upload Whatsapp chat text file", type=["csv", "txt"])
if uploaded_file is not None:
    # Read uploaded file as dataframe
    df= df_data = parse_chat(uploaded_file)

    # Clean and lemmatize
    column = 'Chat'
    df_data = preprocess.call_all(df_data, column )

    # Load tokenizer and model
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Filter words present in tokenizer
    loaded_word_index = tokenizer.word_index

    unknown = []
    def word_present(text):
        s = ""
        for word in text.split(' '):
            if word in loaded_word_index:s += " " + word
            else:unknown.append(word)
        return s 

    df[column] = df[column].apply(lambda x: word_present(x))
    df = df.dropna()

    # Drop rows where 'Chat' is NaN or only contains whitespace
    df = df.dropna(subset=['Chat'])
    df = df[df['Chat'].str.strip() != '']
    df = df.reset_index(drop=True)

    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(df_data['Chat'])
    padded_seq = pad_sequences(sequences, maxlen=40, padding='post', truncating='post')

    # Predict emotions
    model = load_model('model/emotion_analyzer.h5')
    preds = model.predict(padded_seq)
    preds = np.argmax(preds, axis=1)
    df_data = df_data.reset_index(drop=True)
    df_data['predict'] = preds

    # Prepare data for plotting
    unique_names = df_data['Name'].unique()
    chat_emotions = {}
    for name in unique_names:
        indices = df_data.index[df_data['Name'] == name].tolist()
        chat_emotions[name] = df_data.loc[indices, 'predict']

    # Emotion label map
    label_map = {
        0: 'Sadness',
        1: 'Joy',
        2: 'Love',
        3: 'Anger',
        4: 'Fear',
        5: 'Surprise'
    }

    fig2 = plot_function.plot_piechart(chat_emotions, label_map)
    st.pyplot(fig2)

    # Plot chat count
    fig1 = plot_function.plot_chatcount(df_data)
    st.pyplot(fig1)

else:
    st.info("Please upload a Whatsapp TXT file to analyze.")









# import pandas as pd
# import numpy as np
# from nltk.corpus import stopwords
# import re
# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import matplotlib.pyplot as plt
# import streamlit as st
# import pickle
# import numpy as np
# from keras.models import load_model
# import plot_function
# import preprocess

# # read csv file
# df=  pd.read_csv("mabel.txt", header=None, on_bad_lines='skip', encoding='utf8')
# df.head()

# # Make as dataset format

# df=df.drop(0)
# df=df.drop(columns=[0])
# df.columns=['Chat']
# Message=df["Chat"].str.split("-",n=1,expand=True)
# Message1=Message[1].str.split(":",n=1,expand=True)
# df["Name"]=Message1[0]
# df["Chat"]=Message1[1]
# df=df_data =df[["Name","Chat"]]

# # clean data and lemmitization
# column='Chat'
# df = preprocess.call_all(df, column)
# df.head()


# # Load tokenizer
# with open('model/tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# loaded_word_index = tokenizer.word_index
# unknown = []
# def word_present(text):
#     s = ""
#     for word in text.split(' '):
#         if word in loaded_word_index:s += " " + word
#         else:unknown.append(word)
#     return s 
# df[column] = df[column].apply(lambda x: word_present(x))
# df = df.dropna()
# df[column].head()

# sequence = tokenizer.texts_to_sequences(df[column])
# padded_seq = pad_sequences(sequence,maxlen=40, padding='post', truncating= 'post')
# model = load_model('model/emotion_analyzer.h5')
# predict = model.predict(padded_seq)
# predict = np.argmax(predict,axis=1)
# predict =  pd.DataFrame(predict, columns=['predict'])

# df = df.reset_index(drop=True)
# unique_names= df['Name'].unique()
# name_chat_dict = {}
# chat_emotions = {}
# for name in unique_names:
#     name_df = df[df['Name'] == name]
#     name_chat_dict[name] = name_df.index  #name_df['Chat']
#     chat_emotions[name] =  predict['predict'].loc[list(name_df.index)]
# plot_function.plot_chatcount(df_data)

# label_map = {
#     0: 'sadness',
#     1: 'joy',
#     2: 'love',
#     3: 'anger',
#     4: 'fear',
#     5: 'surprise'
# }

# plot_function.plot_piechart(chat_emotions, label_map)