import re
import pandas   as pd
import numpy    as np
from   datetime import datetime, timedelta

# constants
CATEGORIES = {
    'open' : '0',
    'not a real question' : '1',
    'not constructive' : '2',
    'off topic' : '3',
    'too localized' : '4'
}

INTERROGATIVE_WORDS = ['who', 'where', 'when', 'which', 'how', 'why', 'what']


MAX_DATE = datetime(2012, 12, 31, 23, 59, 59)

def interrogative_words(text):
    text = text.lower()
    num  = 0
    for word in INTERROGATIVE_WORDS:
        num += len(re.findall(r'\b{}\b'.format(word), text))
    return num

def is_question(text):
    return 1 if '?' in text else 0

def sentences_with_I(text):
    text  = text.lower()
    items = re.split(r'[.!?]+', text)[:-1]
    sum = 0
    for item in items:
        try:
            if item[0] == ' ':
                compare = item[1]
            else:
                compare = item[0]
            if compare == 'i':
                sum += 1
        except:
            pass
    return sum

def sentences_with_You(text):
    text  = text.lower()
    items = re.split(r'[.!?]+', text)[:-1]
    sum = 0
    for item in items:
        try:
            if item[0] == ' ':
                compare = item[1:4]
            else:
                compare = item[:3]
            if compare == 'you':
                sum += 1
        except:
            pass
    return sum

def number_of_links(text):
    text = text.lower()
    num  = len(re.findall(r'\b{}\b'.format('http'), text))
    return num

def n_sentences(text):
    return len(re.split(r'[.!?]+', text))

def first_sentence_lenght(text):
    return len(re.split(r'[.!?]+', text)[0])

def n_digits(text):
    return len(re.split(r'[0-9]', text))

def upper_letters_ratio(text):
    return len(re.findall(r'[A-Z]',text))/len(text)

def lower_letters_ratio(text):
    return len(re.findall(r'[a-z]',text))/len(text)

# looking in user table
users = pd.read_csv('Users.csv')
users = users.set_index('Id')

def find_user_upvotes(userid):
    try:
        # user = users[users['Id']== userid].iloc[0]
        user = users.loc[userid]
        return user['UpVotes']
    except:
        return 0

def find_user_downvotes(userid):
    try:
        # user = users[users['Id']== userid].iloc[0]
        user = users.loc[userid]
        return user['DownVotes']
    except:
        return 0

def find_user_views(userid):
    try:
        # user = users[users['Id']== userid].iloc[0]
        user = users.loc[userid]
        return user['Views']
    except:
        return 0

final_data = pd.DataFrame(columns=["PostId", "PostAge",
                                  "OwnerAge", "TitleLenght", "BodyLenght", "IsTitleAQuestion",
                                  "InterrogativeWordsInTitle", "InterrogativeWordsInBody",
                                  "NumberBodySentencesWithI", "NumberBodySentencesWithYou",
                                  "NumberBodySentences", "FirstSentenceLenght", "NumberBodyDigits",
                                  "NumberBodyLinks", "UpperLettersRatioInBody", "LowerLettersRatioInBody",
                                  "NumberOfTags","UserUpVotes" ,"UserDownVotes" ,"UserViews"])

chunksize = 200000
for data in pd.read_csv('train-sample.csv', chunksize=chunksize):

    subdata = pd.DataFrame(columns=["PostId", "PostAge",
                                   "OwnerAge", "TitleLenght", "BodyLenght", "IsTitleAQuestion",
                                   "InterrogativeWordsInTitle", "InterrogativeWordsInBody",
                                   "NumberBodySentencesWithI", "NumberBodySentencesWithYou",
                                   "NumberBodySentences", "FirstSentenceLenght", "NumberBodyDigits",
                                   "NumberBodyLinks", "UpperLettersRatioInBody", "LowerLettersRatioInBody",
                                   "NumberOfTags","UserUpVotes","UserDownVotes","UserViews"])

    # updating data types
    data["PostCreationDate"]  = pd.to_datetime(data["PostCreationDate"])
    data["OwnerCreationDate"] = pd.to_datetime(data["OwnerCreationDate"])
    data["BodyMarkdown"]      = data["BodyMarkdown"].astype('str')

    # creating new columns
    subdata["PostId"]   = data["PostId"]
    subdata["PostAge"]  = data["PostCreationDate"].apply(lambda x: (MAX_DATE - x).days)
    subdata["OwnerAge"] = data["OwnerCreationDate"].apply(lambda x: (MAX_DATE - x).days)

    subdata["TitleLenght"] = data["Title"].apply(lambda x: len(x))
    subdata["BodyLenght"]  = data["BodyMarkdown"].apply(lambda x: len(x))

    subdata["IsTitleAQuestion"]          = data["Title"].apply(lambda x: is_question(x))
    subdata["InterrogativeWordsInTitle"] = data["Title"].apply(lambda x: interrogative_words(x))
    subdata["InterrogativeWordsInBody"]  = data["BodyMarkdown"].apply(lambda x: interrogative_words(x))

    subdata['NumberBodySentencesWithI']   = data["BodyMarkdown"].apply(lambda x: sentences_with_I(x))
    subdata['NumberBodySentencesWithYou'] = data["BodyMarkdown"].apply(lambda x: sentences_with_You(x))

    subdata['NumberBodySentences'] = data["BodyMarkdown"].apply(lambda x: n_sentences(x))
    subdata['FirstSentenceLenght'] = data["BodyMarkdown"].apply(lambda x: first_sentence_lenght(x))
    subdata['NumberBodyDigits']    = data["BodyMarkdown"].apply(lambda x: n_digits(x))
    subdata['NumberBodyLinks']     = data["BodyMarkdown"].apply(lambda x: number_of_links(x))

    subdata['UpperLettersRatioInBody'] = data["BodyMarkdown"].apply(lambda x: upper_letters_ratio(x))
    subdata['LowerLettersRatioInBody'] = data["BodyMarkdown"].apply(lambda x: lower_letters_ratio(x))

    subdata['NumberOfTags'] = data[['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']].isnull().sum(axis=1)
    subdata['NumberOfTags'] = subdata["NumberOfTags"].apply(lambda x: 5 - x)

    subdata['UserUpVotes']   = data['OwnerUserId'].apply(lambda x: find_user_upvotes(x))
    subdata['UserDownVotes'] = data['OwnerUserId'].apply(lambda x: find_user_downvotes(x))
    subdata['UserViews']     = data['OwnerUserId'].apply(lambda x: find_user_views(x))

    subdata['OpenStatus'] = data['OpenStatus']

    # categorizing target
    # subdata['OpenStatus'] = data['OpenStatus'].apply(lambda x:  CATEGORIES[x])
    final_data = final_data.append(subdata)

final_data.to_csv('train-sample-preprocessed-categorical.csv', index=False)
