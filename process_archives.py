import collections
import cPickle as pickle
import HTMLParser
import json
import mechanize
import sys
import time

from bs4 import BeautifulSoup

sys.setrecursionlimit(10000)
url = "https://mailman.stanford.edu/pipermail/java-nlp-user/"
url_sub = "https://mailman.stanford.edu/pipermail/java-nlp-user/2016-June/thread.html"


br = mechanize.Browser()
br.open(url)
archives = br.response().read()
soup = BeautifulSoup(archives)

url_suffixes = []

# Iterate through all <a> tags in base page
for a_elem in soup.find_all("a"):
    if a_elem is not None:
        text = a_elem.string
        if text == "[ Thread ]":
            url_suffixes.append(a_elem["href"])


all_quarter_archives = [url + suffix for suffix in url_suffixes]


def process_text(msg_text):
    """
    Clean the msg_text so that there are no trailing messages from thread
    :param msg_text:
    :return:
    """
    newline_split = msg_text.split("\n")

    # Find first ">" line and ignore everything after
    first = -1
    for idx, line in enumerate(newline_split):
        if line[:2] == "> ":
            first = idx
            break

    # TODO: Remove all newlines
    processed = "\n".join(newline_split[:first-2])
    return processed


def process_question(question_url):
    br.open(question_url)
    question_archives = br.response().read()
    question_soup = BeautifulSoup(question_archives)

    pre_tag = question_soup.find("pre")
    text_contents = ""
    if pre_tag is not None:
        text_contents = str(pre_tag.text.encode("utf-8"))
        processed = process_text(text_contents)
    return processed


def process_quarter_archives(archives_url, title_to_thread):
    print "Opening archive url: {0}".format(archives_url)
    br.open(archives_url)
    sub_archives = br.response().read()
    sub_soup = BeautifulSoup(sub_archives)

    title_to_urls = collections.defaultdict(list)

    for li in sub_soup.find_all("li"):
        msg_urls = []
        prefix = li.find("a").string[:15]
        if prefix != "[java-nlp-user]":
            continue
        thread_title = li.find("a").string[16:].strip("\n")
        thread_url = li.find("a")["href"]
        if len(title_to_urls[thread_title]) != 0:
            continue
        msg_urls.append(thread_url)

        for ul in li.find_all("ul"):
            url = ul.find("li").find("a")["href"]
            msg_urls.append(url)
        title_to_urls[thread_title] = msg_urls

    # Remove /thread.html suffix
    base_url = archives_url[:-11]

    # Process all questions in archive
    for title, thread in title_to_urls.iteritems():
        thread_contents = []
        for url in thread:
            question_url = base_url + url
            question_contents = process_question(question_url)
            thread_contents.append(question_contents)
        title_to_thread[title.strip("\n")] = thread_contents
    print "Done processing: {0}".format(archives_url)



# Crawl and scrape all data
title_to_thread = collections.defaultdict(list)
start_time = time.time()

# Process all quarter urls
for a_url in all_quarter_archives:
    process_quarter_archives(a_url, title_to_thread)

print "Total time to process all questions: ", str(time.time() - start_time)

print "Done processing questions. Now JSON-ing..."

# output to JSON
with open("nlp_user_questions.json", "wb") as f:
    json.dump(title_to_thread, f)

