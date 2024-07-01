import re

def getCleanText(txt):
    """
    Clean the text by removing <phrase> tags added by Autophrase library
    @param txt: text to be cleaned
    @return: cleaned text
    """
    txt = txt.replace("</phrase>", "")
    txt = re.sub('<phrase_Q=[0-1].{3}[0-9]>', '', txt)
    return txt


def removeSpecialChars(txt):
    """
    Remove special characters in a text
    @param txt: input text
    @return: cleaned text
    """
    return re.sub(' +', ' ', re.sub('[^a-zA-Z0-9 \n\.]', '', txt))