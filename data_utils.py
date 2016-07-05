import collections
import cPickle as pickle
import json
import re

# TODO: Build glove vecs using corpus of Java code?
# TODO: Generate train/dev/test split of data -- can do that later

def clean_so_text(so_text):
    """
    Clean data from Stack Overflow, removing <p> and <code> and various other tags
    :param so_text:
    :return:
    """
    # TODO: Do I want to remove all newlines? Removing for now...

    str_to_replace = ["<code>", "</code", "<p>", "</p>", "<pre>", "</pre>",
                      "<blockquote>", "</blockquote>", "\n", "<em>", "</em>",
                      "<strong>", "</strong>", "\t"]

    new_str = so_text
    for s in str_to_replace:
        new_str = new_str.replace(s, "")

    return new_str


def extract_text_vocab(text):
    """
    Tokenize text and return a set and list of vocab words
    :param text:
    :return:
    """
    text_tokens = re.findall(r"<|>|[\w]+|,|\?|\.|\(|\)|\\|\"|\/|;|\#|\&|\$|\%|\@|\{|\}|\+|\-|\:", text)
    lower_tokens = [t.lower() for t in text_tokens]
    return set(lower_tokens), lower_tokens



def so_data_statistics(data_file):
    """
    Report statistics such as number of comments/answers/questions for given data
    :param data_file: json of data file
    :return:
    """
    with open(data_file, "r") as f:
        data = json.load(f)

    answer_to_num_questions = collections.Counter()
    comment_to_num_questions = collections.Counter()
    num_comments = 0
    num_answers = 0
    num_questions = len(data)

    for q in data:
        q = json.loads(q)
        q_comments = 0
        q_comments += len(q["comments"])
        q_answers = len(q["answers"])
        for a in q["answers"]:
            q_comments += len(a["comments"])

        answer_to_num_questions[q_answers] += 1
        comment_to_num_questions[q_comments] += 1

        num_comments += q_comments
        num_answers += q_answers

    print "Num comments: {0}, Num answers: {1}, Num_questions: {2}".format(
        num_comments, num_answers, num_questions)
    print "-" * 10
    print "Answers map: ", answer_to_num_questions
    print "Comments map: ", comment_to_num_questions

    return num_comments, num_answers, num_questions, answer_to_num_questions, \
           comment_to_num_questions


def lists_data_statistics(data_file):
    with open(data_file, "rb") as f:
        data = json.load(f)

    answer_to_num_questions = collections.Counter()
    num_questions = len(data)

    for title, thread in data.iteritems():
        num_answers = len(thread)-1
        answer_to_num_questions[num_answers] += 1

    print "Num questions: {0}".format(num_questions)
    print "Answers map: ", answer_to_num_questions


if __name__ == "__main__":
    #so_data_statistics("questions.json")j
    #lists_data_statistics("nlp_user_questions_space.json")
    #get_so_vocab("snlp_so_questions.json", True)
    #get_mailman_vocab("nlp_user_questions_space.json", True)
    #glv_vocab = get_glv_vocab("/Users/mihaileric/Documents/Research/LSTM-NLI/data/glove.6B.50d.txt.gz")

    #print "Len missing SO: ", len(so.difference(glv_vocab))
    #print "Len missing mailman: ", len(mailman.difference(glv_vocab))
    #print "Len missing combined: ", len(combined.difference(glv_vocab))
    _, _, _, so_word_to_idx, mailman_word_to_idx, total_word_to_idx = gen_vocab_file()
    gen_data("data/snlp_so_questions.json", "data/nlp_user_questions_space.json")
    tokenize_data("data/data_sentences.txt", total_word_to_idx)