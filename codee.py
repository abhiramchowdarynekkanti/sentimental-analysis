# ignore warning
import warnings
warnings.filterwarnings('ignore')

# data manipulation
import pandas as pd

# data visulization
import seaborn as sns
import matplotlib.pyplot as plt

# text processing
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem import PorterStemmer

# remove emoji
import emoji

# regular expression
import re

# wordcloud
from wordcloud import WordCloud
from collections import Counter

# model building and evaluation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

27 | P a g e
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score,
confusion_matrix
add Codeadd Markdown
Load Data
add Codeadd Markdown

[ ]:

# load data
cols=['tweetid', 'entity', 'target', 'content']

data0 = pd.read_csv("/kaggle/input/twitter-entity-sentiment-
analysis/twitter_training.csv",names=cols)

data1 = pd.read_csv("/kaggle/input/twitter-entity-sentiment-
analysis/twitter_validation.csv",names=cols)

final_df = pd.concat([data0,data1])
add Codeadd Markdown

[ ]:

final_df.head()
add Codeadd Markdown

[ ]:

# data dimension
final_df.shape
add Codeadd Markdown

[ ]:

# info about data
final_df.info()
add Codeadd Markdown

[ ]:

# check null values
final_df.isna().sum()
add Codeadd Markdown

[ ]:

28 | P a g e
# drop null values
final_df.dropna(inplace=True)
add Codeadd Markdown

[ ]:

# check duplicates values
final_df.duplicated().sum()
add Codeadd Markdown

[ ]:

# drop duplicates
final_df.drop_duplicates(inplace=True)
add Codeadd Markdown

[ ]:

# value count of target col
final_df['target'].value_counts().plot(kind='pie',autopct='%.2f')
plt.title("per count of each target value")
plt.show()
add Codeadd Markdown

[ ]:

target_count = final_df['target'].value_counts().reset_index()
target_count
add Codeadd Markdown

[ ]:

plt.figure(figsize=(15,5))
ax =
sns.barplot(data=target_count,x='target',y='count',palette='cubehelix')
for bars in ax.containers:
ax.bar_label(bars)

plt.title("Count of each target value")
plt.show()
add Codeadd Markdown

[ ]:

# tweet count of each user
tweet_count =
final_df.groupby('tweetid')['target'].count().sort_values(ascending=False).
reset_index()
tweet_count = tweet_count.rename(columns={'target':'count'})

29 | P a g e
tweet_count
add Codeadd Markdown
feature engineering
add Codeadd Markdown

[ ]:

# char count
final_df['char_count'] = final_df['content'].apply(len)
# word count
final_df['word_count'] = final_df['content'].apply(lambda
x:len(nltk.word_tokenize(x)))
# sentence count
final_df['sent_count'] = final_df['content'].apply(lambda
x:len(nltk.sent_tokenize(x)))
add Codeadd Markdown

[ ]:

# dist plot

fig, axes = plt.subplots(1,3,figsize=(18,5))
sns.distplot(ax=axes[0],x=final_df['char_count'],color='b')
axes[0].set_title('char distribution')

sns.distplot(ax=axes[1],x=final_df['word_count'],color='g')
axes[1].set_title('word distribution')

sns.distplot(ax=axes[2],x=final_df['sent_count'],color='r')
axes[2].set_title('sentence distribution')
plt.show()
add Codeadd Markdown

[ ]:

# drop unnecessary cols
final_df = final_df.drop(columns=['tweetid','entity'],axis=1)
add Codeadd Markdown

[ ]:

final_df.head()
add Codeadd Markdown

[ ]:

30 | P a g e
# remove emojis from tweets
final_df['content'] = final_df['content'].apply(lambda x:
emoji.replace_emoji(x,replace=''))