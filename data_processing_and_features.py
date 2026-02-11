import regex as re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Use sklearn built-in stopwords (no download required)
stopwords_list = set(ENGLISH_STOP_WORDS)

exclude = string.punctuation


def remove_html_tags(text):
    reg_rule = re.compile('<.*?>')
    return re.sub(reg_rule, '', text)


def remove_urls(text):
    reg_rule = re.compile(r'http\S+|www.\S+')
    return re.sub(reg_rule, '', text)


def remove_stopwords(text):
    # Faster and cleaner version
    return " ".join(
        word for word in text.split()
        if word not in stopwords_list
    )


def remove_punc(text):
    return text.translate(str.maketrans('', '', exclude))


def text_data_cleaning(df):
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(remove_html_tags)
    df['review'] = df['review'].apply(remove_urls)
    df['review'] = df['review'].apply(remove_punc)
    df['review'] = df['review'].apply(remove_stopwords)
    return df


def tfidf_features_fit(df):
    tfidf = TfidfVectorizer(min_df=0.01, max_df=0.1)
    tfidf_matrix = tfidf.fit_transform(df['review'])
    return tfidf, tfidf_matrix.toarray()


def tfidf_features_transform(tfidf, df):
    tfidf_matrix = tfidf.transform(df['review'])
    return tfidf_matrix.toarray()
