import collections
import cPickle as pickle
import HTMLParser
import json
import mechanize
import sys
import time

from bs4 import BeautifulSoup

sys.setrecursionlimit(10000)
# Base page where links to all quarter archives reside
url = "https://mailman.stanford.edu/pipermail/java-nlp-user/"
url_sub = "https://mailman.stanford.edu/pipermail/java-nlp-user/2016-June/thread.html"


# TODO: Clean out names included in the messages

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
    Do various types of cleaning on message text.
    :param msg_text:
    :return:
    """
    newline_split = msg_text.split("\n")

    # Find first ">" line and ignore everything after, to remove previous messages existing in text
    # of current message
    first = -1
    for idx, line in enumerate(newline_split):
        if line[:2] == "> ":
            first = idx
            break

    # Do more cleaning of messages
    filtered = []
    for line in newline_split:
        line = line.strip(" ")
        # Kind of hacky rule-based cleaning of messages
        if line == "" or line == " " or line[:2] == "--" or line[:4] == "<htt" \
                or line[:3] == "URL" or line[:5] == "<mail" or line[:7] == "An HTML"\
                or line[0] == ">":
            continue
        filtered.append(line.strip(" "))

    space_delimited = " ".join(filtered[:first-2])
    newline_delimited = "\n".join(filtered[:first-2])

    return space_delimited, newline_delimited


def process_question(question_url):
    """
    Get content from a single question thread
    :param question_url:
    :return:
    """
    br.open(question_url)
    question_archives = br.response().read()
    question_soup = BeautifulSoup(question_archives)

    pre_tag = question_soup.find("pre")
    text_contents = ""
    if pre_tag is not None:
        text_contents = str(pre_tag.text.encode("utf-8"))
        processed = process_text(text_contents)
    return processed


def process_quarter_archives(archives_url, title_to_thread_newline , title_to_thread_space):
    """
    Process an entire quarters worth of archives
    :param archives_url:
    :param title_to_thread_newline:
    :param title_to_thread_space:
    :return:
    """
    print "OPENING ARCHIVE URL: {0}".format(archives_url)
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
        thread_contents_newline = []
        thread_contents_space = []
        for url in thread:
            question_url = base_url + url
            space_delim, newline_delim = process_question(question_url)
            thread_contents_newline.append(newline_delim)
            thread_contents_space.append(space_delim)
        title_to_thread_newline[title.strip("\n")] = thread_contents_newline
        title_to_thread_space[title.strip("\n")] = thread_contents_space
    print "Done processing: {0}".format(archives_url)




# Crawl and scrape all data -- maintain two formats:
# 1) where newline delimited messages
# 2) where space delimited messages
title_to_thread_newline = collections.defaultdict(list)
title_to_thread_space = collections.defaultdict(list)
start_time = time.time()

# Process all quarter urls
for a_url in all_quarter_archives:
    process_quarter_archives(a_url, title_to_thread_newline, title_to_thread_space)


print "Total time to process all questions: ", str(time.time() - start_time)
print "Done processing questions. Now JSON-ing..."

# output to JSON
with open("nlp_user_questions_newline.json", "wb") as f:
    json.dump(title_to_thread_newline, f)

with open("nlp_user_questions_space.json", "wb") as f:
    json.dump(title_to_thread_space, f)

