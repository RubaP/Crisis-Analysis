import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import src.util.utils as Util
import src.util.relevancy as Relevancy
import src.util.file as File


def isRelevant(data, relevant_docs, source):
    """
    Determine whether a data entry in the sample is available in the list of relevant docs retrieved by a relevancy
    retrieval algorithm
    @param data: data entry in the sample
    @param relevant_docs: list of documents retrieved as relevant by a relevancy retrieval algorithm
    @param source: source name
    @return: string 'Relevant' if data entry is available in list of relevant documents retrieved, others 'Irrelevant'
    """
    if data['Source'] != source:
        return data['Relevancy']
    else:
        index = data['Original Index']
        if index in relevant_docs.index:
            return 'Relevant'
        else:
            return 'Irrelevant'


def findMetaDataBasedRankingPerformance(sample, SOURCES, documents, seed_documents, root_dir):
    """
    Find performance of metadata-based ranking algorithm
    @param sample: ground truth data
    @param SOURCES: list of source
    @param documents: list of documents
    @param seed_documents: seed documents containing keywords in the query
    @param root_dir: data directory
    """
    results = []

    for discussion_threshold in range(0, 105, 5):
        for popularity_threshold in range(0, 105, 5):
            for w in np.arange(1, 1.5, 0.1):
                print("--------------------------")
                sample['Relevancy'] = 'Irrelevant'
                print(discussion_threshold, popularity_threshold, w)
                relevant_doc_count = 0
                for source in SOURCES:
                    relevant_doc = Relevancy.getMetaDataBasedRelevantDocsBySeedSubjects(documents[source],
                                                                                        seed_documents[source],
                                                                                        discussion_threshold,
                                                                                        popularity_threshold, w)
                    if relevant_doc is not None:
                        sample['Relevancy'] = sample.apply(isRelevant, args=(relevant_doc, source), axis=1)
                        relevant_doc_count += relevant_doc.shape[0]

                weighted = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                           average='weighted')
                individual = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                             average=None, labels=["Relevant", "Irrelevant"])

                score = [relevant_doc_count, discussion_threshold, popularity_threshold, w, weighted[0], weighted[1],
                         weighted[2]]
                score.extend(individual[0])
                score.extend(individual[1])
                score.extend(individual[2])
                print(score)
                results.append(score)

    np.savetxt(File.getRelevancyResultsDir(root_dir) + "relevancy-results-metadata.csv",
               results,
               delimiter=", ",
               fmt='% s',
               header="size,discussion_threshold,popularity_threshold,w,precision-weighted,recall-weighted,F1-weighted,"
                      "precision-relevant,precision-irrelevant,recall-relevant,recall-irrelevant,F1-relevant,"
                      "F1-irrelevant")


def findWordEmbeddingPerformance(documents, query, root_dir):
    """
    Find performance of word embedding based relevancy ranking
    @param documents: list of documents
    @param query: query words e.g. energy crisis or energy policy
    @param root_dir: data directory
    """
    results = []
    document_scores = {}

    for source in SOURCES:
        document_scores[source] = Relevancy.getWordEmbeddingSimilarityScore(documents[source], query)

    for similarity_threshold in np.arange(0, 1, 0.02):
        sample['Relevancy'] = 'Irrelevant'
        relevant_doc_count = 0
        for source in SOURCES:
            rel_docs = document_scores[source][document_scores[source] >= similarity_threshold]
            sample['Relevancy'] = sample.apply(isRelevant, args=(rel_docs, source), axis=1)

            relevant_doc_count += rel_docs.shape[0]
        weighted = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                   average='weighted')
        individual = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                     average=None, labels=["Relevant", "Irrelevant"])

        score = [relevant_doc_count, similarity_threshold, weighted[0], weighted[1], weighted[2]]
        score.extend(individual[0])
        score.extend(individual[1])
        score.extend(individual[2])
        print(score)
        results.append(score)

    np.savetxt(File.getRelevancyResultsDir(root_dir) + "relevancy-results-word-embedding.csv",
               results,
               delimiter=", ",
               fmt='% s',
               header="size,similarity_threshold,precision-weighted,recall-weighted,F1-weighted,precision-relevant,"
                      "precision-irrelevant,recall-relevant,recall-irrelevant,F1-relevant,F1-irrelevant")


def findTFIDFPerformance(documents, query, root_dir):
    """
    Find performance of TF-IDF based relevancy ranking
    @param documents: list of documents
    @param query: query words e.g. energy crisis or energy policy
    @param root_dir: data directory
    """
    results = []
    document_scores = {}

    for features in range(10, 100, 10):
        for source in SOURCES:
            document_scores[source] = Relevancy.getTFIDFSimilarityScores(documents[source], " ".join(query), features)

        for similarity_threshold in np.arange(0, 1, 0.02):
            sample['Relevancy'] = 'Irrelevant'
            relevant_doc_count = 0
            for source in SOURCES:
                rel_docs = document_scores[source][document_scores[source] >= similarity_threshold]
                sample['Relevancy'] = sample.apply(isRelevant, args=(rel_docs, source), axis=1)

                relevant_doc_count += rel_docs.shape[0]
            weighted = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                       average='weighted')
            individual = precision_recall_fscore_support(sample['Label'].to_numpy(), sample['Relevancy'].to_numpy(),
                                                         average=None, labels=["Relevant", "Irrelevant"])

            score = [features, relevant_doc_count, similarity_threshold, weighted[0], weighted[1], weighted[2]]
            score.extend(individual[0])
            score.extend(individual[1])
            score.extend(individual[2])
            print(score)
            results.append(score)

    np.savetxt(File.getRelevancyResultsDir(root_dir) + "Relevancy/relevancy-results-tfidf.csv",
               results,
               delimiter=", ",
               fmt='% s',
               header="features,size,similarity_threshold,precision-weighted,recall-weighted,F1-weighted,"
                      "precision-relevant,precision-irrelevant,recall-relevant,recall-irrelevant,F1-relevant,"
                      "F1-irrelevant")


config = Util.readAllConfig()
root_dir = config["data-directory"]
SOURCES = config["mainstream-sources"]
SEARCH_KEYS = config["mainstream-data-collection"]["keywords"]
sample = pd.read_excel(File.getGroundTruthDataDir(root_dir) + "Relevancy Ground Truth.xlsx")
SEED_KEYS = config["relevancy-ranking"]["seed-keys"]

documents = {}
seed_documents = {}

for source in SOURCES:
    print("Reading - ", source)
    df = pd.read_csv(File.getFilteredDataDir(root_dir) + source + ".csv", index_col=0)
    df.dropna()
    df = df.loc[df.Body.apply(type) == str]
    print("Number of documents: ", df.shape[0])
    df['subjects'] = df['Meta-Subject'].apply(lambda x: Relevancy.getSubjects(x))  # Get list of subjects
    relevant_doc, df = Relevancy.getRelevantSeedDocuments(df, SEED_KEYS)  # Get seed documents containing seed keys

    seed_documents[source] = relevant_doc
    documents[source] = df

findMetaDataBasedRankingPerformance(sample, SOURCES, documents, seed_documents, root_dir)
findWordEmbeddingPerformance(documents, SEED_KEYS, root_dir)
findTFIDFPerformance(documents, SEED_KEYS, root_dir)
