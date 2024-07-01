import re
from nltk.stem import WordNetLemmatizer
import nltk
import src.util.file as File
import numpy as np
import tomotopy as tp
import os
import pandas as pd

lemmatizer = WordNetLemmatizer()
stop_words = ['government', 'uk', 'britain', 'england', 'country']

nltk.download('wordnet')
nltk.download('stopwords')
special_char_regex = re.compile('[,\[@_!#$%^&*()<>?/\|}{~:\]]')


def isTopicValid(topic_label):
    """
    Check whether a topic is valid or not using the topic label
    :param topic_label:
    :return:
    """
    if topic_label == "INVALID":
        return False
    else:
        return True


def processWord(word):
    """
    Process a word by replacing space by - and multiple occurrences of - by a single -
    @param word: word to be processed
    @return: processed word
    """
    word = re.sub('-+', '-', re.sub(' +', '-', lemmatizer.lemmatize(word.lower())))
    return word


def isValid(word):
    """
    Check whether a word is valid to be considered for the vocabulary of a topic model.
    A word is valid if it is not available in stop words, does not contain any special characters and contains more
    than one character.
    @param word: word
    @return: Boolean indicating whether a word is valid or not
    """
    word = word.lower()
    return word not in stop_words and special_char_regex.search(word) == None and len(word) > 1


def getBagofWords(doc):
    """
    Get bag of word representation of a document
    @param doc: single document
    @return: bag of words representation of a document
    """
    result = re.findall(r'(?:<phrase_Q=[0-1].{3}[0-9]>)(.*?)(?:</phrase>)', doc)
    result = [processWord(word) for word in result if isValid(word)]
    return result


def getBagofWordsOfCorpus(documents):
    """
    Get bag of word representation of list of documents
    @param documents: list of documents
    @return: list of bag of words representation
    """
    return [getBagofWords(doc) for doc in documents]


def getTrainedModel(K, bow):
    """
    Get a trained topic model given number of topics and documents
    @param K: number of topics to be learned
    @param bow: list of documents represented as bag of words
    @return: topic model
    """
    print("Initiating the model with k - ", K)
    model = tp.LDAModel(k=K, min_cf=5)
    for doc in bow:
        model.add_doc(doc)

    return model


def getCoherence(model, N):
    """
    Get coherence score of a topic model
    @param model: topic model
    @param N: top N words to be considered while calculating the coherence
    @return: coherence score of a model
    """
    coh = tp.coherence.Coherence(model, coherence='c_v', top_n=N)
    score = coh.get_score()
    return score


def getTopTopicWords(model, N):
    """
    Get Top N words of topics given a topic model
    @param model: topic model
    @param N: number of top words
    @return: 2D list of top words of topics represented by a topic model
    """
    results = []
    K = model.k

    for i in range(0, K):
        words = []
        words_with_prob = model.get_topic_words(i, N)

        for (word, prob) in words_with_prob:
            words.append(word)

        results.append(words)

    return results


def printModelDetails(model):
    """
    Print topic model details including number of docs, topics, coherence and vocabulary stat
    @param model: topic model
    """
    K = model.k
    print('Num docs:', len(model.docs), 'Num words:', model.num_words)
    print("Num of topics - ", K)
    print("Coherence: ", getCoherence(model, 20))
    printVocabularyStat(model.used_vocabs, model.used_vocab_freq)

    for i in range(0, K):
        print("Topic ", i, " - ", model.get_topic_words(i, 10))

    print("Count by topic: ", model.get_count_by_topics())


def printVocabularyStat(vocabulary, frequency):
    """
    Print statistics of a topic model vocabulary
    @param vocabulary: vocabulary of a topic model
    @param frequency: frequency of words in the vocabulary
    """
    size = len(vocabulary)
    print("Vocabulary size: ", size)
    multiword_count = 0

    for word in vocabulary:
        if "-" in word:
            multiword_count += 1
    print("Number of multi-word phrases in vocabulary: ", multiword_count)
    print("Number of unigrams: ", size - multiword_count)

    top_words = [(vocabulary[i] + " - " + str(frequency[i])) for i in range(0, 100)]
    top_words = " , ".join(top_words)
    print("Top words: [", top_words+"]")


def saveModel(model, path, source):
    """
    Save topic model of a source
    @param model: topic model
    @param path: path to be saved
    @param source: source name
    """
    model.save(path + "Model/" + source, True)

    topic_words = getTopTopicWords(model, 25)
    File.saveAnalysisCSV(topic_words, path + "Label/", source, "Topic-words.csv", ",".join(list(map(str, list((range(1,26)))))))


def loadModel(path, source):
    """
    Load topic model of a source
    @param path: path for the topic model
    @param source: source name
    @return: topic model
    """
    model = tp.LDAModel.load(path + source)

    print("Loading LDA model successful")
    printModelDetails(model)
    return model


def checkTopicExistence(path, source):
    """
    Check whether topic model existing for the given source
    @param path: path for topic model
    @param source: source name
    @return: Boolean indicating whether topic model existing or not
    """
    existence = os.path.exists(path + source)
    print("LOG: Topics existence - ", existence)
    return existence


def getTopicLabels(topic_path, source):
    """
    Get Topic labels for a source
    @param topic_path: directory where topic labels are stored
    @param source: source name
    @return: list of topic labels generated by chat-gpt
    """
    topic_names = pd.read_excel(topic_path + source + "-Topic-words.xlsx")
    name_list = topic_names['Topic'].values.tolist()
    return name_list


def generateTopicCoOccurrenceStat(root_dir, sources):
    """
    Generate topic-topic co-occurrence value per source. Number of times two topics were mentioned in the same document
    will be accounted for the co-occurrence value
    @param root_dir: data directory
    @param sources: list of sources
    """
    topic_path = File.getTopicDir(root_dir)
    sa_srl_path = File.getESRDir(root_dir)

    for source in sources:
        print("source: ", source)
        df = pd.read_csv(sa_srl_path + source + "-Topic-Polarity.csv", on_bad_lines='skip')
        K = df['# Topic-ID'].max() + 1
        weights = [[0] * K for i in range(K)]
        doc_groups = df.groupby(['Doc-ID'])

        for doc, group in doc_groups:
            topics = group["# Topic-ID"].unique()

            for i in topics:
                for j in topics:
                    weights[i][j] = weights[i][j] + 1
        print(weights)

        np.savetxt(topic_path + "Weights" + source + "-topic-topic-weights.csv", weights, delimiter=", ")


def getChatGPTQuery(top_N_words):
    """
    Given list of top N words of a topic, generate the topic label query for chatgpt
    :param top_N_words:
    """
    txt = "Topic represented by words ["

    for i in range(1, 21):
        if i < 20:
            txt = txt + str(top_N_words[str(i)]) + ","
        else:
            txt = txt + str(top_N_words[str(i)]) + "]"
    txt += " in max of 5 words"
    print(txt)


def printChatGPTQuery(root_dir, sources):
    """
    Print chat-GPT query to generate topic labels for all the sources
    :param root_dir: data directory
    :param sources: list of sources
    """
    for source in sources:
        print("---------Processing - ", source, "------------")
        df = pd.read_csv(File.getTopicDir(root_dir) + "Label/" + source + "-Topic-words.csv", encoding="utf-8",
                         index_col=False)
        df.apply(lambda x: getChatGPTQuery(x), axis=1)