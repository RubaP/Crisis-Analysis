import re
import json
import pandas as pd
import os

def saveSampleNewspaperAsText(data, root_dir):
    """
    Save newspaper body into a text file given the newspaper data
    @param data: newspaper data
    """
    body = data['Body']
    id = data['Id']

    title = re.sub('[^A-Za-z0-9 ]+', '', data['Title'])

    body = body.replace(". ", ".\n")
    filename = getGroundTruthDataDir(root_dir) + 'sample/['+str(id)+']-'+title

    if len(filename) > 250:
        filename = filename[:250]

    with open(filename + '.txt', 'w') as f:
        f.write(body)


def saveAnalysisJson(data, path, source):
    """
    Save an analysis as JSON file
    @param data: data to be saved
    @param path: path
    @param source: source name
    """
    with open(path + source + '.json', 'w') as f:
        json.dump(data, f)


def saveAnalysisCSV(data, path, source, filename, header):
    """
    Save an analysis to CSV file
    @param data: data to be saved
    @param path: path
    @param source: source name
    @param filename: file name
    @param header: header of CSV file
    """
    pd.DataFrame(data).to_csv(path + source + "-" + filename, header=header.split(","), encoding="utf-8", index=False)


def readTextDocuments(path):
    """
    Read a text document given in the path
    @param path: path for the text file
    @return: list of string, each element in the list represent a row in the text file
    """
    with open(path, "r", encoding="utf-8") as text_file:
        data = text_file.read()

    documents = data.split("\n\n")
    size = len(documents)
    print("Number of documents: ", size)
    return documents


def getMainstreamDataDir(root_dir):
    """
    Get mainstream data directory
    @param root_dir: data directory
    @return: mainstream data directory
    """
    dir = root_dir + "/newspapers"
    checkAndCreateDirectory(dir)
    return dir


def getFilteredDataDir(root_dir):
    """
    Get mainstream filtered data directory
    @param root_dir: data directory
    @return: mainstream filtered data directory
    """
    dir = getMainstreamDataDir(root_dir) + "/Filtered/"
    checkAndCreateDirectory(dir)
    return dir


def getRelevantDataDir(root_dir):
    """
    Get mainstream relevant data directory
    @param root_dir: data directory
    @return: mainstream relevant data directory
    """
    dir = getMainstreamDataDir(root_dir) + "/Relevant/"
    checkAndCreateDirectory(dir)
    return dir


def getOriginalCSVDataDir(root_dir):
    """
    Get mainstream data CSV files directory. This directory stored unfiltered data
    @param root_dir: data directory
    @return: mainstream data CSV files directory
    """
    dir = getMainstreamDataDir(root_dir) + "/CSV/"
    checkAndCreateDirectory(dir)
    return dir


def getRawDataDir(root_dir):
    """
    Get mainstream data RAW file directory. This directory stored raw files collected from Nexis
    @param root_dir: data directory
    @return: mainstream data RAW file directory
    """
    dir = getMainstreamDataDir(root_dir) + "/Raw/"
    checkAndCreateDirectory(dir)
    return dir


def getGroundTruthDataDir(root_dir):
    """
    Get ground truth data directory
    @param root_dir: data directory
    @return: ground truth data directory
    """
    dir = root_dir + "/Ground Truth/"
    checkAndCreateDirectory(dir)
    return dir


def getAutoPhraseDir(root_dir):
    """
    Get Autophrase directory
    @param root_dir: data directory
    @return: Autophrase directory
    """
    dir = root_dir + "/Autophrase/"
    checkAndCreateDirectory(dir)
    return dir


def getResultsDir(root_dir):
    """
    Get topics directory
    @param root_dir: data directory
    @return: Topics directory
    """
    dir = root_dir + "/Results/"
    checkAndCreateDirectory(dir)
    return dir

def getTopicDir(root_dir):
    """
    Get topics directory
    @param root_dir: data directory
    @return: Topics directory
    """
    dir = getResultsDir(root_dir) + "Topics/"
    checkAndCreateDirectory(dir)
    return dir


def getESRDir(root_dir):
    """
    Get Sentiment-SRL directory
    @param root_dir: data directory
    @return: Sentiment-SRL directory
    """
    dir = getResultsDir(root_dir) + "Sentiment-SRL/"
    checkAndCreateDirectory(dir)
    return dir


def getIssueDir(root_dir):
    """
    Get issues directory
    @param root_dir: data directory
    @return: Issues directory
    """
    dir = getResultsDir(root_dir) + "Issues/"
    checkAndCreateDirectory(dir)
    return dir


def getGraphDir(root_dir):
    """
    Get graphs directory
    @param root_dir: data directory
    @return: Graphs directory
    """
    dir = getResultsDir(root_dir) + "Graphs/"
    checkAndCreateDirectory(dir)
    return dir


def getRelevancyResultsDir(root_dir):
    """
    Get relevancy results directory
    @param root_dir: data directory
    @return: Relevancy results directory
    """
    dir = getResultsDir(root_dir) + "Relevancy/"
    checkAndCreateDirectory(dir)
    return dir


def getAnnotationDir(root_dir):
    """
    Get annotation directory
    @param root_dir: data directory
    @return: Annotation directory
    """
    dir = getResultsDir(root_dir) + "Annotation/"
    checkAndCreateDirectory(dir)
    return dir


def checkAndCreateDirectory(dir):
    """
    Create directory
    :param dir: directory path
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
