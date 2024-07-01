import warnings
from sklearn.metrics.pairwise import linear_kernel
import collections
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import nltk
import pandas as pd


warnings.filterwarnings("ignore")
stopwords = set(nltk.corpus.stopwords.words("english"))


def generateTextFile(docs, path, source):
    """
    Generate text file containing body of the relevant newspaper
    @param docs: list of relevant newspaper
    @param path: path to generate text file
    @param source: source name
    """
    text_col = docs['Body'].dropna().to_list()

    text = '\n\n'.join(text_col)
    with open(path + source + "-relevant.txt", "w") as text_file:
        text_file.write(text)


def getSubjects(meta_data):
    """
    Convert subjects available as a string to list of subjects
    @param meta_data: string containing list of subjects
    @return: list of subjects
    """
    subjects = []

    if type(meta_data) is str:
        data_list = meta_data.split(";")
        for data in data_list:
            try:
                weight = int(re.findall(r'\d+', data[-5:])[0])
                subject = data[:-6].strip().lower()
                if "news" not in subject:
                    subjects.append((subject, weight))
            except:
                continue
    return subjects


def getSubjectCounter(subjects):
    """
    Get counter object of list of subjects
    @param subjects: list of subjects
    @return: counter object
    """
    subject_list = []
    for sub_list in subjects:
        subject_list.extend([x[0] for x in sub_list])

    subject_counter = collections.Counter(subject_list)
    print("Total subjects: ", len(subject_counter.keys()))

    return subject_counter


def getRelevantSeedDocuments(docs, keywords):
    """
    Get seed relevant documents using list of keywords. If any of the keyword is presented in the body of a newspaper,
    then the newspaper will be considered as a relevant seed document.
    @param docs: list of documents
    @param keywords: list of keywords
    @return: seed relevant documents
    """
    relevancy = []
    documents = docs['Body']
    for doc in documents:
        relevant = False
        try:
            for key in keywords:
                if key in doc.lower():
                    relevant = True
        except:
            pass
        relevancy.append(relevant)

    relevant_docs = docs.loc[relevancy]
    return relevant_docs, docs


def findNewRelevantDocs(docs, new_subjects, discussion_threshold):
    """
    Find new set of relevant docs according to the new list of relevant subjects and
    @param docs: list of documents
    @param new_subjects: list of new subjects
    @param discussion_threshold: discussion threshold
    @return: list of new relevant documents and remaining documents to be processed for next iteration
    """
    relevant_doc_ids = findNewRelevantDocsIds(docs['subjects'], new_subjects, discussion_threshold)
    relevant_docs = docs.loc[relevant_doc_ids]
    dataset = docs.drop(relevant_docs.index)

    print("Relevant documents size: ", relevant_docs.shape[0])
    print("Remaining dataset size: ", dataset.shape[0])
    return relevant_docs, dataset


def findNewRelevantDocsIds(docs, new_subjects, discussion_threshold):
    """
    Find ids of new set of relevant docs according to the new list of relevant subjects and
    @param docs: list of documents
    @param new_subjects: list of new subjects
    @param discussion_threshold: discussion threshold
    @return: list of ids of new set of relevant documents
    """
    relevancy = []

    for doc in docs:
        relevant = False
        for sub in doc:
            if sub[0] in new_subjects and sub[1] > discussion_threshold:
                relevant = True
                break
        relevancy.append(relevant)
    return relevancy


def findNewRelevantSubjects(subject_counter, doc_size, popularity_threshold, old_relevant_subjects):
    """
    Find new set of relevant subjects
    @param subject_counter: new subject counter
    @param doc_size: number of relevant documents
    @param popularity_threshold: popularity threshold
    @param old_relevant_subjects: old set of relevant subjects
    @return: new set of relevant subjects
    """
    relevant_subjects = []

    probability = {k: v / doc_size for k, v in subject_counter.items()}

    for k, v in probability.items():
        if v >= (popularity_threshold/100):
            relevant_subjects.append(k)
    print("Relevant subjects: ", relevant_subjects)

    relevant_subjects = list(set(relevant_subjects) - set(old_relevant_subjects))
    print("New relevant subjects: ", relevant_subjects)
    print("New relevant subjects count: ", len(relevant_subjects))
    return relevant_subjects


def getMetaDataBasedRanking(docs, discussion_threshold, popularity_threshold, growth_rate, keys):
    """
    Wrapper function for getMetaDataBasedRelevantDocsBySeedSubjects to extract seed documents
    @param docs: list of documents
    @param discussion_threshold: discussion threshold
    @param popularity_threshold: popularity threshold
    @param growth_rate: growth rate controlling discussion threshold
    @param keys: list of search keywords
    @return: list of relevant documents
    """
    docs['subjects'] = docs['Meta-Subject'].apply(lambda x: getSubjects(x))

    relevant_doc, docs = getRelevantSeedDocuments(docs, keys)
    return getMetaDataBasedRelevantDocsBySeedSubjects(docs, relevant_doc, discussion_threshold, popularity_threshold, growth_rate)


def getMetaDataBasedRelevantDocsBySeedDocuments(docs, seed_docs, discussion_threshold, popularity_threshold, growth_rate):
    """
    Get meta-data based relevant docs by starting with seed documents
    @param docs: list of documents
    @param seed_docs: seed documents containing query words
    @param discussion_threshold: discussion threshold
    @param popularity_threshold: popularity threshold
    @param growth_rate: growth rate controlling discussion threshold
    @return: list of relevant documents
    """
    relevant_subjects = []
    new_relevant_subjects = []
    iteration = 0
    relevant_doc = seed_docs

    while iteration == 0 or len(new_relevant_subjects) > 0:
        print("-----------------------")
        threshold = min(discussion_threshold * (pow(growth_rate, iteration)), 100)
        print("Threshold - ", threshold)
        filtered_doc, docs = findNewRelevantDocs(docs, new_relevant_subjects, threshold)

        if relevant_doc is None:
            relevant_doc = filtered_doc
        else:
            relevant_doc = relevant_doc.append(filtered_doc)

        doc_size = relevant_doc.shape[0]
        print("Size of relevant documents: ", doc_size)

        subject_counter = getSubjectCounter(relevant_doc['subjects'])
        new_relevant_subjects = findNewRelevantSubjects(subject_counter, doc_size, popularity_threshold,
                                                        relevant_subjects)
        relevant_subjects.extend(new_relevant_subjects)
        print("Total number of relevant subjects: ", len(relevant_subjects))
        print("Final set of relevant subjects: ", relevant_subjects)
        iteration += 1

    return relevant_doc


def getMetaDataBasedRelevantDocsBySeedSubjects(docs, seed_docs, discussion_threshold, popularity_threshold, growth_rate):
    """
    Get meta-data based relevant docs by starting with seed subjects
    @param docs: documents
    @param seed_docs: seed documents containing query words
    @param discussion_threshold: discussion threshold
    @param popularity_threshold: popularity threshold
    @param growth_rate: growth rate controlling discussion threshold
    @return: list of relevant documents
    """
    relevant_subjects = []
    new_relevant_subjects = []
    iteration = -1
    relevant_doc = None

    while iteration == -1 or len(new_relevant_subjects) > 0:
        print("-----------------------")
        if iteration >= 0:
            threshold = min(discussion_threshold * (pow(growth_rate, iteration)), 100)
            print("Threshold - ", threshold)
            filtered_doc, docs = findNewRelevantDocs(docs, new_relevant_subjects, threshold)

            if relevant_doc is None:
                relevant_doc = filtered_doc
            else:
                relevant_doc = pd.concat([relevant_doc, filtered_doc], ignore_index=True)

            doc_size = relevant_doc.shape[0]
            print("Size of relevant documents: ", doc_size)

            subject_counter = getSubjectCounter(relevant_doc['subjects'])
        else:
            subject_counter = getSubjectCounter(seed_docs['subjects'])
            doc_size = seed_docs.shape[0]

        new_relevant_subjects = findNewRelevantSubjects(subject_counter, doc_size, popularity_threshold,
                                                        relevant_subjects)
        relevant_subjects.extend(new_relevant_subjects)
        print("Total number of relevant subjects: ", len(relevant_subjects))
        print("Final set of relevant subjects: ", relevant_subjects)
        iteration += 1

    return relevant_doc

def getTFIDFSimilarityScores(docs, query, features):
    """
    Get TF-IDF based similarity scores for the documents with respect to the given query
    @param docs: list of documents
    @param query: query string
    @param features: number of features to be considered while generating TF-IDF vector representation of documents
    @return: documents with their similarity score
    """
    documents = docs['Body']
    corpus = documents
    query = " ".join(query)
    vectorizer = TfidfVectorizer(max_features=features, lowercase=True, stop_words=list(stopwords))

    vectors = vectorizer.fit_transform([query] + corpus)
    cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities]  # convert back to native Python dtypes

    docs['similarity_score'] = document_scores
    return docs['similarity_score']


def preprocessForWordEmbedding(doc):
    """
    Preprocess a document for word embedding based ranking
    @param doc: document
    @return: preprocessed document
    """
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]


def getWordEmbeddingSimilarityScore(docs, query):
    """
    Get word embedding based similarity score of documents for the given query
    @param docs: list of documents
    @param query: query string
    @return: documents with their similarity score
    """
    print("=============Word Embedding Ranking============")
    documents = docs['Body']
    corpus = [preprocessForWordEmbedding(document) for document in documents]
    query = preprocessForWordEmbedding(" ".join(query))

    if 'glove' not in locals():  # only load if not already in memory
        glove = api.load("glove-wiki-gigaword-50")

    similarity_index = WordEmbeddingSimilarityIndex(glove)

    dictionary = Dictionary(corpus + [query])
    tfidf = TfidfModel(dictionary=dictionary)

    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)  # , nonzero_limit=None)

    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in corpus]],
        similarity_matrix)

    doc_similarity_scores = index[query_tf]

    docs['similarity_score'] = doc_similarity_scores
    return docs['similarity_score']