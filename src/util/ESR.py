import csv
import spacy
from allennlp.predictors.predictor import Predictor
import re
from NewsSentiment import TargetSentimentClassifier
import numpy as np
import pandas as pd
import src.util.file as File
import src.util.topic as Topic
import src.util.preprocess as Preprocess
import src.util.utils as Util
import src.util.issues as Issue

spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
tsc = TargetSentimentClassifier()

ACTOR_LABELS = ["PERSON", "ORG"]
SEMANTIC_ROLES = ["ARG0", "ARG1", "ARG2", "ARG3", "ARGM-PRP", "ARGM-TMP", "ARGM-LOC", "ARGM-DIR", "ARGM-MNR"]


def getActors(sentence):
    """
    Get list of actors from a sentence. Only the person entities and organization entities will be retrieved as actors
    @param sentence: sentence
    @return: list of actors
    """
    entities = []

    for actor in sentence.ents:
        label = actor.label_
        if label == "PERSON" or label == "ORG":
            entities.append(actor.text)

    return entities


def getSemanticRoleLabels(topicId, sentence, date, topic, docId, senteneId):
    """
    Get semantic role labels present in the sentence
    @param topicId: topic ID
    @param sentence: sentence
    @param date: date of the document
    @param topic: topic label
    @param docId: document ID
    @param senteneId: sentence ID
    @return: list of semantic roles, list of causers
    """
    srl = predictor.predict(
        sentence=sentence[:512])["verbs"]

    semantic_roles = []
    causers = []
    found = False

    for srl_set in srl:
        roles = [topicId, topic, date, docId, senteneId, srl_set["verb"]]
        description = srl_set["description"]

        if "ARG0" in description or "ARG1" in description:
            for ROLE in SEMANTIC_ROLES:
                if ROLE in description:
                    try:
                        term = re.search(r'\[' + ROLE + r': (.*?)\]', description).group(1)
                        roles.append(term)
                        if ROLE == "ARG0":  # ARG0 depicts a causer
                            causers.append(term)
                    except:
                        roles.append("")
                    found = True
                else:
                    roles.append("")

        if found:
            semantic_roles.append(roles)

    return semantic_roles, causers


def getCauserPolarity(sentence, causers):
    """
    Get polarity of causers present in a sentence
    @param sentence: sentence
    @param causers: list of causers
    @return: list of polarity
    """
    polarity_list = []

    for causer in causers:
        text = sentence.split(causer, maxsplit=1)  # Find position of the causer in the sentence
        try:
            # NewSentiment can only handle sentences of length 512 characters. Therefore, 200 characters in the left,
            # and 200 characters in the right are considered for polarity prediction.
            sentiment = tsc.infer_from_text(text[0][-200:0], re.sub('[^A-Za-z0-9]+', '', causer), text[1][0:200])[0]
            polarity = 0

            if sentiment['class_id'] == 0:
                polarity = -1 * sentiment['class_prob']
            elif sentiment['class_id'] == 2:
                polarity = sentiment['class_prob']
            polarity_list.append((causer, polarity))
        except Exception as e:
            print("Error Occurred while annotating sentiment for text: ", sentence, "for the causer: ", causer)
    return polarity_list


def getActorsAndSentiment(sentence):
    """
    Get actors and their corresponding sentence present in a sentence
    @param sentence: sentence
    @return: list of (actor, polarity)
    """
    doc = nlp(sentence)
    results = getActors(doc)

    actor_polarity = []

    for actor in results:
        text = sentence.split(actor, maxsplit=1) # Find position of the actor in the sentence
        try:
            # NewSentiment can only handle sentences of length 512 characters. Therefore, 200 characters in the left,
            # and 200 characters in the right are considered for polarity prediction.
            sentiment = tsc.infer_from_text(text[0][-200:0], re.sub('[^A-Za-z0-9]+', '', actor), text[1][0:200])[0]
            polarity = 0

            if sentiment['class_id'] == 0:
                polarity = -1 * sentiment['class_prob']
            elif sentiment['class_id'] == 2:
                polarity = sentiment['class_prob']
            actor_polarity.append((actor, polarity))
        except:
            print("Error Occurred while annotating sentiment for text: ", sentence, "for actor:", actor)

    return actor_polarity


def saveTopicPolarity(topic_level_analysis, K, topic_labels, source, path):
    """
    Save topic level polarity into a CSV file
    @param topic_level_analysis: topic level polarity details
    @param K: number of topics
    @param topic_labels: list of topic labels
    @param source: source name
    @param path: path for CSV file
    """
    topic_polarity = []

    for i in range(0, K):
        for (date, doc_id, sentence_id, pol) in topic_level_analysis[i]["Polarity-entries"]:
            topic_polarity.append([i, topic_labels[i], date, doc_id, sentence_id, pol])

    File.saveAnalysisCSV(topic_polarity, path, source, "Topic-Polarity.csv",
                         "Topic-ID,Topic,Date,Doc-ID,Sentence-ID,Polarity")


def saveActorPolarity(topic_level_analysis, K, topic_labels, source, path):
    """
    Save actor level polarity into a CSV file
    @param topic_level_analysis: topic level polarity details
    @param K: number of topics
    @param topic_labels: list of topic labels
    @param source: source name
    @param path: path for CSV file
    """
    actor_polarity = []

    for i in range(0, K):

        for (date, doc_id, sentence_id, actor, polarity) in topic_level_analysis[i]["Actor-entries"]:
            actor_polarity.append([i, topic_labels[i], date, doc_id, sentence_id, actor, polarity])

    File.saveAnalysisCSV(actor_polarity, path, source, "Entity-Polarity.csv",
                         "Topic-ID,Topic,Date,Doc-ID,Sentence-ID,Entity,Polarity")


def saveIssuePolarity(topic_level_analysis, K, topic_labels, source, path):
    """
    Save issue level polarity into a CSV file
    @param topic_level_analysis: topic level polarity details
    @param K: number of topics
    @param topic_labels: list of topic labels
    @param source: source name
    @param path: path for CSV file
    """
    issue_polarity = []

    for i in range(0, K):
        for (date, doc_id, sentence_id, issue, polarity) in topic_level_analysis[i]["Issues-entries"]:
            issue_polarity.append([i, topic_labels[i], date, doc_id, sentence_id, issue, polarity])

    File.saveAnalysisCSV(issue_polarity, path, source, "Issue-Polarity.csv",
                         "Topic-ID,Topic,Date,Doc-ID,Sentence-ID,Issue,Polarity")


def identifySentimentEntitiesAndRoles(model, topic_labels, documents, date_published, candidate_issues, source,
                                      annotation_path, sa_srl_path):
    """
    Identify sentiment, entities and roles in list of documents given a topic model
    @param model: topic model
    @param topic_labels: list of topic labels
    @param documents: list of documents
    @param date_published: list of date published
    @param candidate_issues: list of candidate issues of a topic (popular ngrams)
    @param source: source name
    @param annotation_path: annotation path
    @param sa_srl_path: Sentiment-SRL path
    @return: topic_level_analysis as a dictionary of data
    """
    semantic_roles = []

    topic_level_analysis = {}
    num_of_docs = len(documents)

    K = model.k

    for i in range(0, K):
        topic_level_analysis[i] = {  # Generate empty topic level analysis for all the topic
            "Actor-entries": [],
            "Polarity-entries": [],
            "Issues-entries": []
        }

    for d in range(num_of_docs):  # Iterate through the document
        if d % 100 == 0:
            print("processing document: ", d)

        doc = documents[d]
        date = date_published[d]
        sentences = doc.split(". ")  # Split document into sentence
        number_of_sentences = len(sentences)
        for s in range(number_of_sentences):  # Process at sentence level
            sentence = sentences[s]
            bow = Topic.getBagofWords(sentence)  # Convert a sentence into bag of words

            if len(bow) > 0:
                sent = model.make_doc(bow)
                topics = model.infer(sent)  # Predict the topic distribution associated with a sentence
                topic_id = np.array(topics[0]).argmax()  # Find the most probable topic for a sentence

                clean_text = Preprocess.getCleanText(sentence)  # Clean the original sentence
                processed_text = Preprocess.removeSpecialChars(clean_text)
                # Get issues present in a text
                issues_present = Issue.getIssuesPresentInText(bow, candidate_issues[topic_id])
                #Get semantic roles and causers present in a text
                roles, causers = getSemanticRoleLabels(topic_id, processed_text, date, topic_labels[topic_id], d, s)

                if len(roles) > 0:
                    semantic_roles.extend(roles)

                causer_polarity_list = getCauserPolarity(processed_text, causers)  # Get polarity of causers
                actors_and_polarities = getActorsAndSentiment(clean_text)  # Get polarity of actors

                # Update the topic level results
                topic_level_analysis[topic_id] = updateTopicLevelResults(topic_level_analysis[topic_id], issues_present,
                                                                         actors_and_polarities, causer_polarity_list,
                                                                         date, d, s)
                #Save sentence-level annotation
                saveSentenceLevelAnnotation(annotation_path, source, topic_id, topic_labels[topic_id],
                                            [date, d, s, clean_text,
                                           ",".join(issues_present),
                                           ",".join(str(p) for p in causer_polarity_list),
                                           ",".join(e + ":" + str(p) for (e, p) in actors_and_polarities)])

    # Save semantic role labels
    File.saveAnalysisCSV(semantic_roles, sa_srl_path, source, "Semantic Roles.csv",
                         "Topic-ID,Topic,Date,Doc-ID,Sentence-ID,Verb,ARG0,ARG1,ARG2,ARG3,ARGM-PRP,ARGM-TMP,ARGM-LOC,ARGM-DIR,ARGM-MNR")

    return topic_level_analysis


def updateTopicLevelResults(analysis, issues, entities_and_polarities, causer_polarity_list,
                            date, doc_id, sentence_id):
    """
    Update a sentence-level results into topic level analysis
    @param analysis: topic level analysis
    @param issues: list of issues presented in a sentence
    @param entities_and_polarities: list of entity, polarity
    @param causer_polarity_list:  list of causer, polarity
    @param date: date of the document
    @param doc_id: document id
    @param sentence_id: sentence id
    @return: updated analysis
    """
    topic_polarity = 0

    for (actor, polarity) in entities_and_polarities:  # Actor level polarity does not contribute to topic polarity
        analysis["Actor-entries"].append((date, doc_id, sentence_id, actor, polarity))
        topic_polarity += polarity

    if len(causer_polarity_list) > 0:  # Each Causer level polarity is added as polarity found at topic and issue level
        for (causer, polarity) in causer_polarity_list:
            analysis["Polarity-entries"].append((date, doc_id, sentence_id, polarity))

            for issue in issues:  # Polarity will be added to each issue found in the sentence
                analysis["Issues-entries"].append((date, doc_id, sentence_id, issue, polarity))
    return analysis


def extractActorsSentimentAndRoles(root_dir, sources):
    """
    Wrapper for identifying actors, sentiment and roles
    @param root_dir: data directory
    @param sources: list of sources
    """
    relevant_data_dir = File.getRelevantDataDir(root_dir)
    phrase_path = File.getAutoPhraseDir(root_dir)
    topic_model_path = File.getTopicDir(root_dir) + "Model/"
    sa_srl_path = File.getESRDir(root_dir)
    annotation_path = File.getAnnotationDir(root_dir)

    for source in sources:
        print("---------Processing - ", source, "------------")
        # Read relevant documents
        df = pd.read_csv(relevant_data_dir + source + "-relevant.csv", index_col=0)
        date_published = df['Date'].values.tolist()
        # Read documents with quality phrases tagged by Autophrase
        documents = File.readTextDocuments(phrase_path + source + "/" + "segmentation.txt")

        print("LOG: Topic extraction started at ", Util.getCurrentTime())

        # Check whether the topic model exists and read topic model and topic labels
        if Topic.checkTopicExistence(topic_model_path, source):
            model = Topic.loadModel(topic_model_path, source)
            topic_labels = Topic.getTopicLabels(File.getTopicDir(root_dir) + "Label/", source)
        else:
            print("ERROR: Model not available")
            continue

        print("LOG: Started extracting entities, sentiment & roles at: ", Util.getCurrentTime())

        K = model.k

        topic_words = Topic.getTopTopicWords(model, 100)  # Get top N words of a topic
        candidate_issues = Issue.getCandidateIssues(topic_words)  # Find ngrams in top N words
        createSentenceLevelAnnotationFiles(annotation_path, source, topic_labels)
        # extract sentiments, entities and roles in all the documents
        topic_level_analysis = identifySentimentEntitiesAndRoles(model, topic_labels, documents, date_published,
                                                                 candidate_issues, source, annotation_path, sa_srl_path)
        # Save polarity results
        saveTopicPolarity(topic_level_analysis, K, topic_labels, source, sa_srl_path)
        saveActorPolarity(topic_level_analysis, K, topic_labels, source, sa_srl_path)
        saveIssuePolarity(topic_level_analysis, K, topic_labels, source, sa_srl_path)

        print("LOG: Completed extracting topic, actor and sentiment at: ", Util.getCurrentTime())


def createSentenceLevelAnnotationFiles(path, source, topic_labels):
    """
    Create sentence level annotation file of  a source. One annotation file will be created per topic
    @param path: path for annotation files
    @param source: name of the source
    @param topic_labels: list of topic labels
    """
    folder = path + source + "/"

    File.checkAndCreateDirectory(path)
    File.checkAndCreateDirectory(folder)

    for i in range(len(topic_labels)):
        with open(folder + "[" + str(i) + "] " + topic_labels[i] + ".csv", "w", encoding="utf-8", newline='') as file:
            write = csv.writer(file)
            write.writerow(["Date", "Doc-ID", "Sentence-Id", "Sentence", "Issues", "Causer-Polarity", "Actor-Polarity"])


def saveSentenceLevelAnnotation(path, source, topic_id, topic, data):
    """
    Save sentence level annotation
    @param path: path for annotation file
    @param source: source name
    @param topic_id: topic ID
    @param topic: topic label
    @param data: sentence level annotation data
    """
    filename = path + source + "/" + "[" + str(topic_id) + "] " + topic + ".csv"

    with open(filename, 'a', encoding="utf-8", newline='') as file:
        write = csv.writer(file)
        write.writerow(data)
