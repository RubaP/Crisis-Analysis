import json
import time


def readAllConfig():
    """
    Read config file
    @return: config
    """
    with open('../../config.json') as config_file:
        config = json.load(config_file)
    return config


def readConfig(component):
    """
    Read specific component in the config file
    @param component: component from the config file to be read
    @return: component in the config file
    """
    with open('../../config.json') as config_file:
        config = json.load(config_file)
    return config[component]


def getCurrentTime():
    """
    Get current system time
    @return: current system time
    """
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def getSources(main_stream):
    """
    Get all the list of sources to analyse
    :param main_stream: list of mainstream sources
    :return: list of sources
    """
    sources = main_stream.copy()

    return sources
