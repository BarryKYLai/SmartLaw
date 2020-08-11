# -*- coding: utf-8 -*-

import os
import jieba
import json
import word2vec
import numpy as np
import pandas as pd

# 最後輸出：pd.DataFrame(train_list, columns=["text", "category"])
'''train.append([train_padding, victory or defeat])'''

def wordseg_and_dfset(path, input): # 一篇一篇斷詞、存進dataframe
    replace_str = ["\"", "\\", "r", "n", " ", "　"]
    stopwords = stopwords_input()
    count = 0
    def_or_not = ""
    train_list = []
    for judgement_indx in input:
        with open(
             path + "\\" + judgement_indx, 
             mode="r", 
             encoding="utf-8"
        ) as input_text:
            data = []
            json_read = json.load(input_text)
            judgement = json.dumps(json_read["judgement"], ensure_ascii=False)
            maintext = json.dumps(json_read["mainText"], ensure_ascii=False)
            opinion = json.dumps(json_read["opinion"], ensure_ascii=False)
            for indx in replace_str:
                judgement = judgement.replace(indx, "")
                maintext = maintext.replace(indx, "")
                opinion = opinion.replace(indx, "")
            dot = maintext.find('。')
            def_or_not = maintext[:dot+1]                
            judgement_cut = jieba.cut(
                            judgement + maintext + opinion, 
                            cut_all=False
            )
            judgement_cut = list(
                            filter(lambda i: i not in stopwords, judgement_cut)
            )
            for indx in judgement_cut:
                if indx[0] == '0' or indx[0] == '1' \
                    or indx[0] == '2' or indx[0] == '3' \
                    or indx[0] == '4' or indx[0] == '5' \
                    or indx[0] == '6' or indx[0] == '7' \
                    or indx[0] == '8' or indx[0] == '9':
                    print('', end='')
                else:
                    count = count+1
                    print("已成功進行", count, "筆斷詞")
                    data.append(indx)
            numb = 256 - (len(data)%256)
            if numb != 0:
                for indx in range(numb):
                    data.append(0)
            data = np.array(data).reshape(-1, 256)
            # 將斷詞結果組成長度為256的句子
            
            for indx in range(data.shape[0]): # 總共有多少句
                listdata = data[indx][:].tolist()
                # print(listdata)
                if not def_or_not.find("駁回") == -1: # 如果是駁回
                    train_list.append([listdata, category_num['defeat']])
                    print([i for i in train_list], "測試用字串..........")
                else:
                    train_list.append([listdata, category_num['victory']])
   
    return train_list

def stopwords_input():
    stopwords = []
    with open(
         "E:\\KuiYou\\others\\reference\\stopwords_new.txt", 
         mode="r", 
         encoding="utf-8"
    ) as stopword_text:
        for indx in stopword_text:
            indx = indx.strip()
            stopwords.append(indx)
        return stopwords

# main:

category_num = {'victory':0, 'defeat':1}

try:
    jieba.load_userdict(
          "E:\\KuiYou\\others\\reference\\text_for_Jieba_new.txt"
    )
    print("匯入斷詞資料庫成功")
except:
    print("失敗，請檢察您的檔名及路徑")

judgement_path = "E:\\KuiYou\\data\\Intellectual-property-court-new-train"
judgement_test = "E:\\KuiYou\\user-input-test"
# 一次匯入所有資料集
judgement_inputs = [judgement_name for judgement_name in os.listdir(judgement_test) if os.path.isfile(os.path.join(judgement_test, judgement_name))]
# 用os.listdir以及list comperhension找出目錄底下所有檔案名稱

judgement_segment = wordseg_and_dfset(judgement_test, judgement_inputs)
train_df = pd.DataFrame(judgement_segment, columns=['text', 'category'])
np.savetxt(
   'E:\\KuiYou\\outputs\\train_df_new_train.txt', 
   train_df.values, 
   fmt='%s %s', # 字串格式
   encoding="utf-8"
)

train_df.to_pickle("E:\\KuiYou\\outputs\\train_df_new_train.pkl") # 儲存為pickle檔