from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import datetime
import src.util.utils as Util


def initiateProcess(login_to_nexis):
    """
    Initiate the entire download process
    @param login_to_nexis: boolean indicate whether the process should login to Nexis first. If the variable is false,
    directly the newspaper search page will be loaded.
    """
    global login_failure_count

    if login_failure_count == 3:  # If the login attempt fails three times, kill the process
        print("ERROR: Login Failed Thrice. Terminating the Process")
        exit(0)

    if login_to_nexis:
        clearCache()
    driver.get(url="https://gold.idm.oclc.org/login?url=https://advance.lexis.com/nexis?&identityprofileid=23X3NH60188")
    time.sleep(50)
    if login_to_nexis:
        LoginToNexis()
    searchPapers()
    filterSourcePublication()
    chooseFiles()


def initiateURL(port):
    """
    initiate the Nexis login URL page in chrome
    @param port: chrome port
    """
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:" + port)

    chrome_driver = "C:\Drivers\ChromeDriver\chromedriver.exe"
    driver_instance = webdriver.Chrome(chrome_driver, chrome_options=chrome_options)
    print("LOG: Chrome Driver Initiated at ", Util.getCurrentTime())
    return driver_instance


def LoginToNexis():
    """
    Login to Nexis by entering username, password and clicking on the login button
    """
    global login_failure_count
    try:
        try:
            email = driver.find_element(By.NAME, 'loginfmt')
        except:
            initiateProcess(True)
        email.send_keys(config['username'])
        doubleClickAndSleep('//*[@id="idSIButton9"]', 5)  # click submit button

        email = driver.find_element(By.NAME, 'passwd')
        email.send_keys(config['password'])
        clickNavigationAndSleep('//*[@id="idSIButton9"]', 5)  # click submit button
        print("LOG: Nexis Login Submitted at ", Util.getCurrentTime())
        time.sleep(120)
    except Exception as e:
        print(e)
        print("ERROR: Login Failed. Reinitiating the process")
        login_failure_count += 1
        initiateProcess(login_to_nexis=True)


def clearCache():
    """
    Clear chrome cache to initiate a new Nexis session
    """
    driver.execute_cdp_cmd('Storage.clearDataForOrigin', {
        "origin": '*',
        "storageTypes": 'all',
    })


def generateSearchQuery(keywords):
    """
    Generate search query by combing list of keywords
    @param keywords: list of keywords to be searched
    @return: search query
    """
    search_query = ''

    for keyword in keywords:
        if len(search_query) == 0:
            search_query += '"' + keyword + '"'
        else:
            search_query += ' or "' + keyword + '"'


def searchPapers():
    """
    Search for newspapers by proving starting and ending date and search query
    """
    global login_failure_count, end_date

    try:
        clickAndSleep('//*[@id="ndbhkkk"]/ln-gns-searchbox/lng-searchbox/div[2]/button[2]', 30)
        login_failure_count = 0

        search_bar = driver.find_element(By.XPATH, '//*[@id="searchTerms"]')

        while search_bar.get_attribute('value') == '':
            search_bar.send_keys(generateSearchQuery(config['keywords']))

        from_date = driver.find_element(By.XPATH, '//*[@id="dateFrom"]')
        while from_date.get_attribute('value') == '' or from_date.get_attribute('value') != start_date:
            from_date.clear()
            from_date.send_keys(start_date)

        to_date = driver.find_element(By.XPATH, '//*[@id="dateTo"]')
        while to_date.get_attribute('value') == '' or to_date.get_attribute('value') != end_date:
            to_date.clear()
            to_date.send_keys(end_date)

        print("LOG: Search Key and Dates Entered at ", Util.getCurrentTime())
        clickNavigationAndSleep('//*[@id="ndth9kk"]/div/a/footer/span/button[1]', 30)
    except Exception as e:
        print(e)
        print("ERROR: Enable to search papers. Reinitiating the process")
        login_failure_count += 1
        initiateProcess(login_to_nexis=True)


def filterSourcePublication():
    """
    Filter source publication by expanding sources tray and clicking on the specific source name
    """
    try:
        publication_list = driver.find_element(By.XPATH, '//*[@id="podfiltersbuttonsource"]')

        while publication_list.get_attribute("class") == "icon trigger la-TriangleRight  collapsed":
            print("LOG: Expanding Publication Sources")
            publication_list.click()

        time.sleep(20)
        clickAndSleep('//*[@id="refine"]/ul[5]/li[6]/button', 10)  # Click more
        for i in range(1, 200):
            publication = driver.find_element(By.XPATH, '//*[@id="refine"]/ul[5]/li['+str(i)+']/label')
            if publication.text.startswith(SOURCE_LABEL):
                filtered = False

                while not filtered:
                    publication.click()
                    time.sleep(20)
                    try:
                        driver.find_element(By.XPATH, '//*[@id="sidebar"]/div[1]/div[2]/ul/li[2]/button')
                        filtered = True
                    except:
                        pass
                break

        print("LOG: Source Publication Filtered ", Util.getCurrentTime())

        clickAndSleep('//*[@id="content"]/header/div[3]/div/button', 10)  # click Group duplicates
        clickAndSleep('//*[@id="content"]/header/div[3]/div/aside/ul/li[2]/button', 20)  # select high similarity

        clickAndSleep('//*[@id="select"]', 10)  # Click sort by
        clickAndSleep('//*[@id="select"]/option[4]', 20)  # Click newest to oldest document
    except:
        print("ERROR: Enable filter papers. Reinitiating the process")
        initiateProcess(login_to_nexis=False)


def checkFileExistence(filename):
    """
    Check whether the download file is existing to make sure the download is successful
    @param filename: file name
    """
    isfile = os.path.isfile(DOWNLOAD_FOLDER + filename)
    print("LOG: Checking file existence - ", isfile)
    return isfile


def clickAndSleep(xpath, sleeping_time):
    """
    Click on the element represented by the xpath and sleep for a certain time
    @param xpath: xpath of a html element
    @param sleeping_time: sleeping time
    """
    driver.find_element(By.XPATH, xpath).click()
    time.sleep(sleeping_time)


def clickNavigationAndSleep(xpath, sleeping_time):
    """
    Click on a navigation element represented by xpath and sleep for a certain time
    @param xpath: xpath of a html element
    @param sleeping_time: sleeping time
    """
    old_url = driver.current_url
    new_url = old_url
    navigation = driver.find_element(By.XPATH, xpath)

    while old_url == new_url:
        navigation.click()
        time.sleep(10)
        new_url = driver.current_url
    time.sleep(sleeping_time)


def doubleClickAndSleep(xpath, sleeping_time):
    """
    Double-click on the element represented by the xpath and sleep for a certain time
    @param xpath: xpath of a html element
    @param sleeping_time: sleeping time
    """
    navigation = driver.find_element(By.XPATH, xpath)
    navigation.click()
    try:
        navigation.click()  # clikcing twice
    except:
        pass
    time.sleep(sleeping_time)


def click(xpath):
    """
    Click and element represented by the xpath
    @param xpath: xpath of a html element
    """
    driver.find_element(By.XPATH, xpath).click()


def download_files(file_count, global_id):
    """
    Download selected files by providing file name and formatting
    @param file_count: number of files download so far
    @param global_id: an id generated for each round of download per session
    """
    success = False
    retried_count = 0

    while not success:
        try:
            # Click Download
            clickAndSleep('//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[3]/ul/li[3]/button', 30)

            filename = WebDriverWait(driver, 60).until(
                EC.visibility_of_element_located((By.XPATH, '//*[@id="FileName"]')))
            filename.clear()
            name = FILE_PREFIX + "-" + str(global_id) + "-" + str(file_count * 10)
            filename.send_keys(name)

            clickAndSleep('//*[@id="Rtf"]', 3)  # select RTF file type
            clickAndSleep('//*[@id="SeparateFiles"]', 3)  # select seperate file option

            clickAndSleep('//*[@id="tab-ContentSpecificOptions"]', 3)  # click content-specific option

            meta_data_checkbox = driver.find_element(By.XPATH, '//*[@id="IncludeSmartIndexingTerms"]')

            if not meta_data_checkbox.is_selected():
                meta_data_checkbox.click()

            click('/html/body/aside/footer/div/button[1]')  # click submit
            print("LOG: Download initiated at ", Util.getCurrentTime())
            time.sleep(360)

            if checkFileExistence(name + ".zip"):
                success = True
        except:
            print("ERROR: Failed to download at ", Util.getCurrentTime(), " with retry count - ", retried_count)
            time.sleep(600)
            retried_count += 1

            if retried_count == 5:
                initiateProcess(login_to_nexis=True)


def clear_selections():
    """
    Clear the newspapers selected for download for the next round of download. This is done by clicking on the clear
    delivery tray
    """
    clear_tray_button = driver.find_element(By.XPATH,
                                            '//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[2]/div/button')

    while clear_tray_button.get_attribute("aria-expanded") == "false":
        clickAndSleep('//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[2]/div/button/span[3]', 20)  # Click tray

    clickAndSleep('//*[@id="viewtray-dropdown"]/div/div[1]/div/button[2]', 10)  # click clear all
    clickAndSleep('/html/body/aside/footer/div/button[1]', 10)  # click clear delivery tray


def updateEndDate():
    """
    Every time a set of newspapers are downloaded, updated the new end data
    """
    global end_date
    date = driver.find_element(By.XPATH, '//*[@id="bisnexis-flex"]/div[1]/ul/li[1]/div/a').text
    end_date = datetime.datetime.strptime(date, '%d %b %Y').strftime('%d/%m/%Y')
    print("LOG: end date for the current download - ", end_date)


def chooseFiles():
    """
    Click and select files iteratively by clicking on the checkbox and next page number. Once given number of pages are
    selected, the downloading process will be initiated.
    """
    global global_id, end_date
    print("LOG: Started downloading from ", start_date, " to ", end_date, " at ", Util.getCurrentTime())
    page_id = 1

    global_id += 1
    while True:
        time.sleep(10)

        if page_id % PAGE_COUNT == 1:
            updateEndDate()
        try:
            select_all = driver.find_element(By.XPATH, '//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[1]/input')
        except:
            print("ERROR: Checkbox clicked while refreshing")
            time.sleep(10)
            select_all = driver.find_element(By.XPATH, '//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[1]/input')

        try:
            waited_count = 0
            while select_all.is_selected():
                time.sleep(10)
                waited_count += 1

                if waited_count == 2:
                    next_page = driver.find_element(By.XPATH, '//*[@id="content"]/div[2]/form/div[2]/nav/ol/li['
                                                    + next_page_id + ']/a')
                    next_page.click()
                    print("LOG: Re-clicked next page, ", page_id)
                    time.sleep(10)
                print("LOG: Waiting for next page")
        except:
            print("ERROR: Checkbox verified while refreshing")
            time.sleep(10)
            select_all = driver.find_element(By.XPATH, '//*[@id="results-list-delivery-toolbar"]/div/ul[1]/li[1]/input')

        select_all.click()
        time.sleep(10)

        if page_id % PAGE_COUNT == 0:
            download_files(page_id, global_id)
            clear_selections()

        next_page_id = str(min(7, page_id + 2))
        try:
            clickAndSleep('//*[@id="content"]/div[2]/form/div[2]/nav/ol/li['
                          + next_page_id + ']/a', 3)  # click next page
        except:
            download_files(page_id, global_id)  # next page does not exist
            break
        page_id += 1


config = Util.readConfig("mainstream-data-collection")
FILE_PREFIX = config["file-prefix"]
CHROME_PORT = config["port"]
SOURCE_LABEL = config["source-label"]
DOWNLOAD_FOLDER = config["download-folder"]
PAGE_COUNT = config["page-count"]

start_date = config["start-date"]
end_date = config["end-date"]
login_to_nexis = config["login-to-nexis"]
login_failure_count = 0
global_id = 0

driver = initiateURL(CHROME_PORT)
initiateProcess(login_to_nexis)
