#!/usr/bin/env python
# coding: utf-8

# In[377]:


import pandas as pd
from nltk.corpus import wordnet 
import wordsegment
from wordsegment import segment
import spacy
import re


# In[382]:


nlp = spacy.load('en')


# In[2]:


url = '/Users/FQ/Downloads/finalDataset.csv'
df1 = pd.read_csv(url)


# In[3]:


syns = wordnet.synsets("program")


# In[4]:


w1 = wordnet.synset('run.v.01') # v here denotes the tag verb 
w2 = wordnet.synset('sprint.v.01') 
print(w1.wup_similarity(w2)) 


# In[5]:


#This cell returns error since hashtags can not be recognized by wordnet as they are not words. 
w1 = wordnet.synset(df1["Hashtag"][1]+".n.01")
w2 = wordnet.synset(df1["Hashtag"][2]+".n.01")
print(w1.wup_similarity(w2)) 


# In[77]:


#importing all categories into memory
cats = list(set(df1["Category"]))


# In[375]:


#removing &, /, , 
for i in range(len(cats)):
    cats[i] = re.sub(r"\/", " ", cats[i])
    cats[i] = re.sub("\&", "", cats[i])
    cats[i] = re.sub("\,", "", cats[i])

cats = [cat.lower() for cat in cats]    


# In[379]:


spacy download en_vectors_web_lg


# In[381]:


nlp = spacy.load('en_core_web_lg')
#nlp = spacy.load('en_vectors_web_lg')


# In[23]:


from sklearn.metrics.pairwise import cosine_similarity


# In[383]:


#Vectorization is different for upper and lower case
w1 = nlp("TV Media Entertainment")
w2 = nlp("TV")
w3 = nlp("Media")
w4 = nlp("Entertainment")
vec1 = w1.vector
vec2 = w2.vector
vec3 = (w2.vector+w3.vector+w4.vector)
print("distance between \"TV\" and \"TV Media Entertainment\":",cosine_similarity(vec2.reshape(1,len(vec2)),vec1.reshape(1,len(vec1))))
print("distance between \"TV\" and \"TV\"+ \"Media\"+ \"Entertainment\":",cosine_similarity(vec2.reshape(1,len(vec2)),vec3.reshape(1,len(vec3))))
w1 = nlp("TV Media Entertainment".lower())
w2 = nlp("TV".lower())
w3 = nlp("Media".lower())
w4 = nlp("Entertainment".lower())
vec1 = w1.vector
vec2 = w2.vector
vec3 = (w2.vector+w3.vector+w4.vector)
print("distance between \"tv\" and \"tv media entertainment\":",cosine_similarity(vec2.reshape(1,len(vec2)),vec1.reshape(1,len(vec1))))
print("distance between \"tv\" and \"tv\"+ \"media\"+ \"entertainment\":",cosine_similarity(vec2.reshape(1,len(vec2)),vec3.reshape(1,len(vec3))))


# In[79]:


#the lesson is in the case of compound word it is better to consider the sum of vectors
split_cats = [cat.split(" ") for cat in cats]


# In[107]:


def f(x):
    vec = np.zeros(384)
    for word in x:
        if word!= "": 
            tok_word = nlp(word)
            vec+=tok_word.vector  
    return vec


# In[109]:


split_cats_vec = list(map(lambda x: f(x), split_cats))


# In[69]:


from nltk.cluster import KMeansClusterer
import nltk


# In[130]:


NUM_CLUSTERS = 30
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters= True)
assigned_clusters = kclusterer.cluster(split_cats_vec, assign_clusters=True)
print(assigned_clusters)


# In[131]:


cls = [[] for i in range(NUM_CLUSTERS)]
for i, cls_num in enumerate(assigned_clusters):  
    cls[cls_num].append(cats[i])


# In[132]:


cls


# In[315]:


dict_cat={}
for i in range(69):
    for j in range(i+1,70):
        u = split_cats_vec[i]
        v = split_cats_vec[j]
        dist = cosine_similarity(u.reshape(1,384),v.reshape(1,384))
        if dist >= 0.8:
            dict_cat[(i,j)] = 1
        else:
            dict_cat[(i,j)] = 0


# In[317]:


l = np.where(np.array(list(dict_cat.values()))==1)
keys = list(dict_cat.keys())
for i in l[0]:
    print(cats[keys[i][0]], " ",cats[keys[i][1]]) 


# In[137]:


from wordsegment import load, segment
load()


# In[133]:


tags = df1["Hashtag"]
len(tags)


# In[139]:


split_tags = list(map(lambda x: segment(x), tags))


# In[141]:


split_tags_vec = list(map(lambda x: f(x), split_tags))


# In[142]:


#np.save('split_tags_vec.npy', split_tags_vec)
#tst = np.load('split_tags_vec.npy')


# In[ ]:





# In[289]:


NUM_CLUSTERS = 30
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters= True)
assigned_clusters = kclusterer.cluster(split_cats_vec, assign_clusters=True)


# In[293]:


len(split_tags_vec)


# In[294]:


dict_tag={}
for i in range(63433):
    for j in range(i+1,63434):
        u = split_tags_vec[i]
        v = split_tags_vec[j]
        dist = cosine_similarity(u.reshape(1,384),v.reshape(1,384))
        if dist >= 0.8:
            dict_tag[(i,j)] = 1
        else:
            dict_tag[(i,j)] = 0
            


# In[306]:


u = split_tags_vec[38]
v = split_tags_vec[20712]
cosine_similarity(u.reshape(1,384),v.reshape(1,384))


# In[309]:


split_tags[38]
split_tags[20712]


# In[303]:


keys = list(dict_tag.keys())
list(dict_tag.values())


# In[254]:


l = np.where(np.array(list(dict_tag.values()))==1)
keys = list(dic.keys())
for i in l[0]:
    print(cats[keys[i][0]], " ",cats[keys[i][1]]) 


# In[177]:


u=split_tags_vec[1]
v=split_tags_vec[2]
cosine_similarity(v.reshape(1,384),u.reshape(1,384))


# In[155]:


cls = [[] for i in range(NUM_CLUSTERS)]
for i, cls_num in enumerate(assigned_clusters):  
    cls[cls_num].append(df1["Category"][i])


# In[325]:


#import collections
d = df1.groupby("Category")["Hashtag"].apply(lambda x: list(x))


# In[352]:


df_cat2tag = pd.DataFrame()


# In[354]:


df_cat2tag["Category"] = ['Advertising', 'Agency', 'Animals', 'Apps', 'Art', 'Aviation', 'Barber',
       'Beauty', 'Cars', 'Clothing', 'Craft', 'Ecommerce', 'Electronics',
       'Events', 'Fashion', 'Feng shui', 'Film/ Cinema', 'Film/Cinema',
       'Finance', 'Financial Services', 'Fishing', 'Fitness', 'Food',
       'Food/ Resturants', 'Food/Resturants', 'FundRaising', 'Furniture',
       'General', 'Home Decor', 'Influencer', 'Inspirational', 'Kids', 'LOVE',
       'Legal/Law', 'Life Coaching', 'Luxury', 'Marketing', 'Medical',
       'Modelling', 'Music', 'Nature', 'Nutrition', 'Painting', 'Personal',
       'Pets', 'Photography/ Videography', 'Photography/Videography',
       'PodCast', 'Poetry', 'Political', 'Psychology', 'RealEstate',
       'Realstate', 'Religious', 'Small Business', 'Smartphone & Gadgets',
       'Sports', 'TV, Media, Entertainment', 'Tattoos', 'Technology', 'Travel',
       'Vaping', 'Wedding', 'Wine', 'Writing', 'Yoga', 'interieur', 'jewelry',
       'meme', 'weddings']


# In[360]:


df_cat2tag["tags"] = list(d.values)


# In[363]:


df_cat2tag["n_tags"] = [len(x) for x in list(d.values)]


# In[364]:


df_cat2tag


# In[4]:


get_ipython().system('pip install bert-serving-server')


# In[5]:


get_ipython().system('pip install bert-serving-client ')


# In[27]:


f = open("/Users/FQ/DownloadS/uncased_L-12_H-768_A-12/bert_config.json", "r")
print(f.read)


# In[35]:


get_ipython().system('bert-serving-start -model_dir="/Users/FQ/Downloads/uncased_L-12_H-768_A-12/" -num_worker=4 ')
from bert_serving.client import BertClient
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])


# In[93]:


from bert_serving.client import BertClient
bc = BertClient()
sent_encod = bc.encode(['I feel terrible', 'It is awesome day', 'I am sick'])


# In[96]:


cosine_similarity(sent_encod[0].reshape(1,-1),sent_encod[1].reshape(1,-1))


# In[70]:


import pandas as pd
import re


# In[56]:


url = '/Users/FQ/Downloads/finalDataset.csv'
df1 = pd.read_csv(url)


# In[59]:


cats = list(set(df1["Category"]))


# In[71]:


for i in range(len(cats)):
    cats[i] = re.sub(r"\/", " ", cats[i])
    cats[i] = re.sub("\&", "", cats[i])
    cats[i] = re.sub("\,", "", cats[i])

cats = [cat.lower() for cat in cats]    


# In[73]:


cats_encod = bc.encode(cats)


# In[62]:


from nltk.cluster import KMeansClusterer
import nltk


# In[80]:


NUM_CLUSTERS = 35
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters= True)
assigned_clusters = kclusterer.cluster(cats_encod, assign_clusters=True)


# In[81]:


cls = [[] for i in range(NUM_CLUSTERS)]
for i, cls_num in enumerate(assigned_clusters):  
    cls[cls_num].append(cats[i])


# In[82]:


cls


# In[99]:


bc.close()


# In[ ]:




