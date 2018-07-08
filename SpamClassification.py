
# coding: utf-8

# In[1]:


import nltk
#nltk.download_shell()
messages=[line.rstrip() for line in open('C:/Users/Home/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Natural-Language-Processing/smsspamcollection/SMSSpamCollection')]
messages[0]
len(messages)


# In[2]:


for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[3]:


import pandas as pd
messages=pd.read_csv('C:/Users/Home/Desktop/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Natural-Language-Processing/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])
messages.head()
messages.describe()


# In[4]:


messages.groupby('label').describe()


# In[6]:


messages['length']=messages['message'].apply(len)


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


messages['length'].plot.hist(bins=150)


# In[9]:


messages[messages['length']>900]['message'].iloc[0]


# In[11]:


messages.hist(column='length',by='label',bins=100,figsize=(12,4))


# In[12]:


from nltk.corpus import stopwords
print(stopwords.words('english'))


# In[14]:


import string

def text_process(mess):
    '''
    Inputs a string of text and performs following activities:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Clean the text and return
    '''
    nopunc=[char for char in mess if char not in string.punctuation]
    
    nopunc=''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[16]:


messages['message'].head().apply(text_process)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))


# In[18]:


message4=messages['message'][3]
print(message4)


# In[21]:


bow4=bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

bow_transformer.get_feature_names()[4068]
bow_transformer.get_feature_names()[9554]


# In[23]:


messages_bow=bow_transformer.transform(messages['message'])


# In[24]:


print('Shape of sparse matrix:',messages_bow.shape)


# In[25]:


print('Amount of non zero entries:',messages_bow.nnz)


# In[27]:


#Amount of non zero messages by total messages
sparsity=(100*messages_bow.nnz)/(messages_bow.shape[0]*messages_bow.shape[1])
print('sparsity: {}'.format(sparsity))


# In[29]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)


# In[32]:


tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)


# In[33]:


print(tfidf4.shape)


# In[34]:


#Inverse document frequency of any word say 'university'
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[35]:


messages_tfidf=tfidf_transformer.transform(messages_bow)


# In[36]:


#Now we need to train the model using Naive Bayes classification
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['label'])


# In[38]:


spam_detect_model.predict(tfidf4)[0]


# In[39]:


prediction=spam_detect_model.predict(messages_tfidf)
prediction


# In[45]:


from sklearn.cross_validation import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)


# In[49]:


label_train


# In[54]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline1=Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
        ])
pipeline2=Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',RandomForestClassifier())
        ])
      


# In[55]:


pipeline1.fit(msg_train,label_train)
pipeline2.fit(msg_train,label_train)


# In[56]:


predictions1=pipeline1.predict(msg_test)
predictions2=pipeline2.predict(msg_test)


# In[57]:


from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(label_test,predictions1))
print(confusion_matrix(label_test,predictions1))


# In[58]:


print(classification_report(label_test,predictions2))
print(confusion_matrix(label_test,predictions2))

