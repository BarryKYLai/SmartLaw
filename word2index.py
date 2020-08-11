# -*- coding: utf-8 -*-

"""這支程式主要是利用詞向量(bin檔)生成權重矩陣、字典及索引，以供我們訓練網路時使用"""

import json
import os
import jieba
import word2vec
import numpy as np
import pandas as pd
import codecs
from keras.models import Sequential, Model
from keras.layers import Embedding

def wordvectors_convert(words_data):
    judgement_wordvectors = word2vec.load(
        "E:\\KuiYou\\others\\reference\\Word2Vec300.bin"
    )
    words_dictionary = {}
    count = 0
    n = 0
    for indx in words_data:
        try:
            wordvec = judgement_wordvectors[indx.strip()]
            if not wordvec == "":
                words_dictionary.setdefault(indx.strip(), wordvec)
                n = n + 1
            print("字典目前建置中...")
            print("已成功建置了", n, "筆資料")
        except:
            count = count + 1
            print("有", count, "筆資料建置失敗...")
    return words_dictionary

def words_seg(path, input):
    replace_str = ["\"", "\\", "r", "n", " ", "　"] # 先設定好斷詞後要去掉哪些詞
    stopwords = stopwords_input()
    data = []
    count = 0
    for  judgement_indx in input:
        with open(
            path + "\\" + judgement_indx, 
            mode="r", 
            encoding="utf-8"
        ) as input_text:
            json_read = json.load(input_text) # 使用語法讀取json
            judgement = json.dumps(json_read["judgement"], ensure_ascii=False)
            maintext = json.dumps(json_read["mainText"], ensure_ascii=False)
            opinion = json.dumps(json_read["opinion"], ensure_ascii=False)
            for indx in replace_str:
                judgement = judgement.replace(indx, '')
                maintext = maintext.replace(indx, '')
                opinion = opinion.replace(indx, '')
            judgement_cut = jieba.cut(
                            judgement + maintext + opinion, 
                            cut_all=False
            )
            judgement_cut = list(
                            filter(lambda i: i not in stopwords, judgement_cut)
            )
            count = count + 1
            for indx in judgement_cut:
                if indx[0] == "0" or indx[0] == "1" \
                    or indx[0] == "2" or indx[0] == "3" \
                    or indx[0] == "4" or indx[0] == "5" \
                    or indx[0] == "6" or indx[0] == "7" \
                    or indx[0] == "8" or indx[0] == "9":
                    print('', end='')
                else:
                    count = count + 1
                    print("已成功進行", count, "筆斷詞")
                    data.append(indx)
    return data

def stopwords_input(): # 匯入停用詞
    stopwords = []
    with open(
         "E:\\KuiYou\\others\\reference\\stopwords_new.txt", 
         mode="r", 
         encoding="utf-8"
    ) as stopword_text:
        for indx in stopword_text.readlines():
            indx = indx.strip()
            stopwords.append(indx)
    return stopwords

# main:

try:
    jieba.load_userdict("E:\\KuiYou\\others\\reference\\text_for_Jieba_new.txt")
    print("匯入斷詞資料庫成功")
except:
    print("失敗，請檢察您的檔名及路徑")

judgement_path = "E:\\KuiYou\\intellectual-property-court-testdata" # 一次匯入所有資料集
judgement_inputs = [judgement_name for judgement_name in os.listdir(judgement_path) if os.path.isfile(os.path.join(judgement_path, judgement_name))]
# 用os.listdir以及list comperhension找出目錄底下所有檔案名稱

judgement_segment = words_seg(judgement_path, judgement_inputs)
word_dictionary = wordvectors_convert(judgement_segment)
weight_matrix = np.zeros((len(word_dictionary.items()) + 1, 300))
# 字典全向量矩陣
# np語法一定要兩層括號
word_index = {} # 字典全索引
vocab_list = [(word, word_dictionary[word]) \
              for word, _ in word_dictionary.items()
]
for i, vocab in enumerate(vocab_list):
 # enumerate語法可以return (index, key)形態（沒有value）
    word, vec = vocab
    weight_matrix[i + 1] = vec
    word_index[word] = i + 1 # word位置的index = i + 1
    # i不從0開始是因為0是padding代表數

outputs_path = "E:\\KuiYou\\outputs"
path_list = ["word_segment.txt", "word_dictionary.txt", 
             "word_matrix.txt", "word_index.txt"
]

with open(outputs_path + "\\word_segment.txt", 
              mode="w", 
              encoding="utf-8"
    ) as text:
    for indx in judgement_segment:
        text.writelines(str(indx) + " / ")

with open(outputs_path + "\\word_dictionary.txt", 
            mode="w", 
            encoding="utf-8"
) as text:
    text.writelines(str(word_dictionary))

with open(outputs_path + "\\word_matrix.txt", 
        mode="w", 
        encoding="UTF-8"
) as text:
    for indx in weight_matrix:
        text.writelines(str(indx)+", ")

with open(outputs_path + "\\word_index.txt", 
        mode="w", 
        encoding="UTF-8"
) as text:
        text.writelines(str(word_index))