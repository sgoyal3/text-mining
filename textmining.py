### Data Source: Reddit

"""
I used reddit for my text-mining analysis as it's a platform I'm intamtely familair with and use quite often
"""
# nltk.download('popular')
# nltk.download('vader_lexicon')
from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
from nltk.tokenize import word_tokenize, RegexpTokenizer

import praw
import pandas as pd 
import datetime as dt


reddit = praw.Reddit(client_id='yh6oSbGy33As-A', \
                     client_secret='E7AVlrBdcc_7XmOQuBVDkQX6mhE', \
                     user_agent='classwork scraper 1.0 by /u/AmbientHunter', \
                     username='AmbientHunter ', \
                     password='alfaromeo16')
                     

subreddit = reddit.subreddit('NBA')

top_subreddit = subreddit.top()
for submission in subreddit.top(limit=1):
    print(submission.title, submission.id)

headlines = set()
for submission in reddit.subreddit('NBA').new(limit=350):
    headlines.add(submission.title)
    display.clear_output()
    print(len(headlines))

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()
df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)

print(df.label.value_counts())

print(df.label.value_counts(normalize=True) * 100)

#Plotting these numbers counts out
fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

#Processing the data and removing punctuation, unecessary words, etc. 
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)
    
    return tokens

#Finding Word Distributions 

#positive terms
pos_lines = list(df[df.label == 1].headline)

pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

pos_freq.most_common(20)

#negative terms
neg_lines = list(df2[df2.label == -1].headline)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(20)