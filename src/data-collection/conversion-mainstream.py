import os
from striprtf.striprtf import rtf_to_text
from datetime import datetime
import re
import numpy as np
import pandas as pd
import zipfile
import src.util.utils as Util
import src.util.file as File


def getDate(txt, source):
    """
    Get the date of the newspaper from textual content in an RTF file
    @param txt: string containing the date
    @param source: source name
    @return: date
    """
    if source == "Guardian":
        try:
            date = datetime.strptime(txt, '%B %d, %Y %A %I:%M %p %Z')
        except:
            try:
                date = datetime.strptime(txt, '%B %d, %Y %A')
            except:
                try:
                    date = datetime.strptime(txt, '%Y-%m-%d %I:%M %p %Z')
                except:
                    return None
    elif "Online" in source:
        try:
            date = datetime.strptime(txt, '%B %d, %Y %A %I:%M %p %Z')
        except:
            try:
                date = datetime.strptime(txt, '%I:%M %p %Z')
            except:
                try:
                    date = datetime.strptime(txt, '%B %d, %Y %A')
                except:
                    try:
                        date = datetime.strptime(txt, '%I:%M %p EST')
                    except:
                        try:
                            date = datetime.strptime(txt, '%B %d, %Y %A %I:%M %p EST')
                        except:
                            return -1
    else:
        date = datetime.strptime(txt, '%B %d, %Y %A')
    return date.strftime("%Y-%m-%d")


def getLength(txt_list):
    """
    Get the length of the newspaper calculated using number of words in the body
    @param txt_list: list of textual content in an RTF file. Each element in the list represent a line in the file
    @return: numer of words in the newspaper body
    """
    for txt in txt_list:
        if txt.startswith("Length:"):
            return int(re.search(r'\d+', txt).group())


def getSection(txt_list):
    """
    Get the section of the newspaper from textual content in an RTF file
    @param txt_list: list of textual content in an RTF file. Each element in the list represent a line in the file
    @return: section name
    """
    for txt in txt_list:
        if txt.startswith("Section:"):
            section = txt.split("Section:")[1].strip().split("Pg")[0].strip()
            return section
        if txt == "Body":
            return ""


def getWriter(txt_list):
    """
    Get the writer of the newspaper from textual content in an RTF file
    @param txt_list: list of textual content in an RTF file. Each element in the list represent a line in the file
    @return: writer's name
    """
    for txt in txt_list:
        if txt.startswith("Byline"):
            if txt.startswith("Byline: BY "):
                return txt.split("Byline: BY ")[1]
            else:
                return txt.split("Byline: ")[1]


def getBody(txt_list):
    """
    Extract the body of the newspaper from textual content in an RTF file
    @param txt_list: list of textual content in an RTF file. Each element in the list represent a line in the file
    @return: newspaper body as list. Each element in the list represent a line in the RTF file
    """
    body = ''
    for i, txt in enumerate(txt_list):
        if txt.startswith("Body"):
            for j in range(i+3, len(txt_list)):
                txt = txt_list[j]
                if txt == '' or txt == 'Graphic' or txt == 'Classification' or txt == '© Daily Mail':
                    return body, j
                else:
                    body += " " + txt.strip()
    return body, -1


def getMetaData(txt_list):
    """
    Extract required meta-data from the textual content in an RTF file
    @param txt_list: list of textual content in an RTF file. Each element in the list represent a line in the file
    @return: meta-data
    """
    meta_data_types = ["Subject", "Company", "Industry", "Person", "Geographic"]
    meta_data = []

    for mtype in meta_data_types:
        found = False
        for txt in txt_list:
            if txt.startswith(mtype+":"):
                meta_data.append(txt.split(mtype+":")[1].strip())
                found = True
                break
        if not found:
            meta_data.append('')

    return meta_data


def getData(content, source):
    """
    Extract required details from text available in an RTF file of a newspaper
    @param content: text available in an RTF file of a newspaper
    @param source: source name
    @return: list of details extracted (e.g. title, body, published data, section, etc.)
    """
    try:
        text = rtf_to_text(content).replace(u'\xa0', u' ')
    except:
        print("LOG: RTF conversion failed")
        return
    txt_list = text.strip().split('\n')

    title = txt_list[0]
    date = getDate(txt_list[2], source)

    if date == -1:
        date = getDate(txt_list[3], source)

    if not date:
        return

    section = getSection(txt_list)
    length = getLength(txt_list)
    writer = getWriter(txt_list)
    (body, end_index) = getBody(txt_list)

    if end_index < 0:
        return

    data = [title, date, section, length, writer, body]
    data.extend(getMetaData(txt_list[end_index:]))
    return data


def saveCSVFile(dataset, root_dir, source):
    """
    Save list of newspaper data into a CSV file
    @param dataset: newspaper data
    @param root_dir: data directory
    @param source: source name
    """
    arr = np.asarray(dataset)
    dataframe = pd.DataFrame(arr, columns=['Title', 'Date', 'Section', 'Length', 'Writer', 'Body',
                                           "Meta-Subject", "Meta-Company", "Meta-Industry", "Meta-Person",
                                           "Meta-Geographic"])
    dataframe_unique = dataframe.drop_duplicates(subset=['Title', 'Date'], keep='last')
    dataframe_unique.to_csv(File.getOriginalCSVDataDir(root_dir) + source + '.csv')


def isDuplicate(file):
    """
    Check whether a file is duplicating by checking for (x) pattern at the end
    @param file: file name
    @return: Boolean indicating whether a file is a duplicate or not
    """
    return re.search("\([0-9]\).RTF$", file)


def readRTFFiles(root_dir, source):
    """
    Read RTF files per source and convert them to required data format
    @param root_dir: data directory
    @param source: list of sources
    @return: list of formatted newspaper data
    """
    dataset = []
    for root, dirs, zipfiles in os.walk(File.getRawDataDir(root_dir) + source):
        for zfile in zipfiles:
            print("Reading", zfile)
            rel_file = os.path.join(root, zfile)

            with zipfile.ZipFile(rel_file, "r") as zif:
                files = zif.namelist()
                for file in files:
                    if "doclist" in file:
                        continue
                    if isDuplicate(file):
                        continue
                    content = zif.open(file).read().decode('utf-8')
                    data = getData(content, source)
                    if data:
                        dataset.append(data)

    return dataset


def convertFiles(root_dir, sources):
    """
    Convert newspapers in RTF format to CSV file format
    @param root_dir: data directory
    @param sources: list of sources
    """
    for source in sources:
        print("LOG: ", source, "file conversion started at ", Util.getCurrentTime())
        news_data = readRTFFiles(root_dir, source)
        saveCSVFile(news_data, root_dir, source)
        print("LOG: ", source, "file conversion completed at ", Util.getCurrentTime())


config = Util.readAllConfig()
SOURCES = config["mainstream-sources"]

convertFiles(config["data-directory"], SOURCES)
