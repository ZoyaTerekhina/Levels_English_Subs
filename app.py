import streamlit as st
import pickle
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import glob
import os
import os.path
import pandas as pd
import numpy as np
import pysrt
import re
import chardet
import string
import pymorphy2

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import dump

from functools import reduce


DICTIONARY = "./English_level/English_level/Oxford_CEFR_level/dictionary.xlsx"
#ENGLISH_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
def load_words():
    with open("./df_words",
              "rb") as fid_1:
        return pickle.load(fid_1)
#df_words = load_words()

def load_model():
    with open("./model_finish", "rb") as fid:
        return pickle.load(fid)

model = load_model()

# функция очистки субтитров от лишних символов:
# Константы:
HTML = r'<.*?>'  # html тэги меняем на пробел
TAG = r'{.*?}'  # тэги меняем на пробел
COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]'  # комменты в скобках меняем на пробел
UPPER = r'[[A-Za-z ]+[\:\]]'  # указания на того кто говорит (BOBBY:)
LETTERS = r'[^a-zA-Z\'.,!? ]'  # все что не буквы меняем на пробел
SPACES = r'([ ])\1+'  # повторяющиеся пробелы меняем на один пробел
DOTS = r'[\.]+'  # многоточие меняем на точку
SYMB = r"[^\w\d'\s]"  # знаки препинания кроме апострофа


def clean_subs(sentence):
    sentence = re.sub(r'\n', ' ', sentence)
    sentence = re.sub(HTML, ' ', sentence)  # html тэги меняем на пробел
    sentence = re.sub(TAG, ' ', sentence)  # тэги меняем на пробел
    sentence = re.sub(COMMENTS, ' ', sentence)  # комменты в скобках меняем на пробел
    sentence = re.sub(UPPER, ' ', sentence)  # указания на того кто говорит (BOBBY:)
    sentence = re.sub(LETTERS, ' ', sentence)  # все что не буквы меняем на пробел
    sentence = re.sub(DOTS, r'.', sentence)  # многоточие меняем на точку
    sentence = re.sub(SPACES, r'\1', sentence)  # повторяющиеся пробелы меняем на один пробел
    sentence = re.sub(SYMB, '', sentence)  # знаки препинания кроме апострофа на пустую строку
    sentence = re.sub('www', '', sentence)  # кое-где остаётся www, то же меняем на пустую строку
    sentence = re.sub(r'(\ufeff)?\d+\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\t?\d{1,3}:\d{1,2}:\d{1,2},\d{1,5}\t?', '',
                      sentence)  # Удаление временной метки
    sentence = sentence.lstrip()  # обрезка пробелов в начале и в конце
    sentence = sentence.encode('ascii', 'ignore').decode()  # удаляем все что не ascii символы
    sentence = sentence.lower()  # текст в нижний регистр
    return sentence


def read_sub(subs):
    text = []
    for i in subs:
        sentence = clean_subs(i.text_without_tags)
        text.append(sentence)
    return ' '.join(text)


def tokenize(column):
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]


# функция чтения файлов из общего каталога
def read_file(dirname, filename):
    fullpath = os.path.join(dirname, filename)
    return read_file(fullpath)

def read_file(fullpath):
    try:
        enc = chardet.detect(open(fullpath, "rb").read())['encoding']
        subs = pysrt.open(fullpath, enc)
    except Exception as e:
        st.write(e)
        st.write('файл не читается', fullpath)
        return False
    return read_sub(subs)

# название и картинка
col1, col2 = st.columns(2)
from PIL import Image

image = Image.open('./eng_level.jpg')
col1.image(image)

col2.title(':green[Фильмы на английском с удовольствием]')
st.subheader('Определим уровень английского языка, которому соответствует выбранный фильм, по субтитрам')
st.write ()
st.write ('Выберите фильм с английскими субтитрами на любом доступном сайте видеопроката. Скачайте файл с субтитрами к фильму в формате .srt. Загрузите файл, используя окно ниже')

uploaded_file = st.file_uploader('Загрузите файл с субтитрами в формате .srt', type='.srt', key=None)

# если файл загружен, то декодируем его и через pysrt открываем

if uploaded_file is not None:
    try:
        encoding = chardet.detect(uploaded_file.getvalue())['encoding']
        subs = pysrt.from_string(uploaded_file.getvalue().decode(encoding))
        filename = uploaded_file.name
        #subs = read_file(filename)
        #data = pd.DataFrame({'movie': uploaded_file.name, 'subtitels': content})

    except:
        st.error(f'К сожалению, не удалось распознать файл')
    subsText = read_sub(subs)
data = pd.DataFrame({'movie': filename, 'subtitels': subsText}, index=[0])

st.write(data)

# считываем словарь и создаём дополнительный объект
df_words = pd.read_excel(DICTIONARY)
file_replace = {'American_Oxford_3000_by_CEFR_level.pdf':'USA', 'American_Oxford_5000_by_CEFR_level.pdf':'USA', 'The_Oxford_3000_by_CEFR_level.pdf':'Oxford',
'The_Oxford_5000_by_CEFR_level.pdf':'Oxford'}
df_words['file'] = df_words['file'].replace(file_replace, regex=True)
df_words = df_words.drop_duplicates(subset=['word'], keep='last')

#st.write(df_words)

col21, col22, col23 = st.columns(3)
if col22.button('Нажмите для получения информации'):
#txt = st.text("Процесс обработки данных занимает некоторое время. Пожалуйста,подождите...")



   english_stopwords = stopwords.words('english')

   prep_text = [tokenize(text) for text in data['subtitels'].astype('str')]

   data['subs_prep'] = prep_text


   def token_stopwords(text):
      tokens_without_sw = [word for word in text if not word in english_stopwords]
      return tokens_without_sw


           # выполняем нормализацию данных
   data['subs_nomal'] = data['subs_prep'].apply(token_stopwords)
   dict_words = {}
       # создаём новые колонки по уровням в основном датасете:
   for level in df_words['level'].unique():
       data[level] = 0
       dict_words[level] = df_words.loc[df_words['level'] == level, 'word'].values


       # Устанавливаем доли слов определённых категорий в фильме, используя словарь с указанием уровня:
   def level_words(row):
       words = row['subs_lemm']
       for level in df_words['level'].unique():
           row[level] = len([word for word in words if word.lower() in dict_words[level]]) / len(words)
       return row


           # Лемантизация:
   morph = pymorphy2.MorphAnalyzer()
   lemm_texts_list_1 = []
   lemm_texts_list_2 = []
   for text in data['subs_nomal']:
       text_lem = [morph.parse(word)[0].normal_form for word in text]
       if len(text_lem) <= 1:
           lemm_texts_list_1.append('')
           lemm_texts_list_2.append('')
           continue
       lemm_texts_list_1.append(text_lem)
       lemm_texts_list_2.append(' '.join(text_lem))
   data['subs_lemm'] = lemm_texts_list_1
   data = data[data['subs_lemm'] != '']
   data = data.apply(level_words, axis=1)
   data['sub_for_ml'] = lemm_texts_list_2
   data = data[data['sub_for_ml'] != '']
   #st.write(data)
   features = data[['sub_for_ml', 'A1', 'A2', 'B1', 'B2', 'C1']]
   #st.write(features)
   pred = model.predict(features)

   st.markdown(f'Требуемый уровень знаний английского языка для просмотра выбранного фильма: {pred[0]}')

   st.write('В системе CEFR знания и умения учащихся подразделяются на три крупных категории, которые далее делятся на шесть уровней:')
   st.text('A1 Уровень - Beginner (Начальный)')
   st.text('A2 Уровень - Elementary (Базовый)')
   st.text('B1 Уровень - Pre-Intermediate (Средний)')
   st.text('B2 Уровень - Upper-Intermediate (Выше среднего)')
   st.text('C1 Уровень - Advanced (Продвинутый)')
   st.text('C2 Уровень - Proficiency (Владение в совершенстве)')



    

