import collections
import json
import re

# TODO: Build glove vecs using corpus of Java code?
# TODO: Generate train/dev/test split of data
def output_to_data_format(data_file, split=[0.8, 0.1, 0.1]):
    """
    Output data to desired format (i.e. src utterance -> tgt utterance).
    Also generate a train/dev/test split according to given split ratio.
    :param split:
    :return:
    """

def get_glv_vocab(glv_file):
    """
    Output set with all tokens in GloVe vocabulary
    :param glv_file:
    :return:
    """
    glv_vocab = set()
    with open(glv_file) as f:
        for line in f:
            contents = line.split(" ")
            glv_vocab.add(contents[0])

    print "Len GloVe vocab: ", len(glv_vocab)
    return glv_vocab


def extract_text_vocab(text):
    """
    Tokenize text and return a set of vocab words
    :param text:
    :return:
    """
    text_tokens = re.findall(r"<|>|[\w]+|,|\?|\.|\(|\)|\\|\"|\/|;|\#|\&|\$|\%|\@|\{|\}|\+|\-|\:", text)
    return set([t.lower() for t in text_tokens])


def get_so_vocab(data_file, skip_no_answer=False):
    """
    Iterate through all text of SO data, tokenize, and generate a list
    of the vocabulary.
    :param data_file:
    :return:
    """
    with open(data_file, "rb") as f:
        data = json.load(f)

    vocab = set()
    for question in data:
        # TODO: Whether to include text of question title?
        question = json.loads(question)

        q_body = question["body"]
        answers = question["answers"]
        comments = question["comments"]

        if skip_no_answer:
            # There is no dialogue because no comments/answers to question
            if len(answers) == 0 and len(comments) == 0:
                continue

        # Extract vocab from question body
        body_vocab = extract_text_vocab(q_body)
        vocab.update(body_vocab)

        # Extract vocab from question comments
        for c in comments:
            vocab.update(extract_text_vocab(c))

        # Extract vocab from question answers and answer comments
        for a in answers:
            a_text = a["text"]
            vocab.update(extract_text_vocab(a_text))
            a_comments = a["comments"]
            for a_c in a_comments:
                vocab.update(extract_text_vocab(a_c))

    return vocab



def get_mailman_vocab(data_file, skip_no_answer=False):
    """
    Iterate through all text of mailman data, tokenize, and generate a list
    of the unique vocabulary tokens.
    :param data_file:
    :return:
    """
    with open(data_file, "rb") as f:
        data = json.load(f)

    vocab = set()

    # TODO: Whether to process title for vocab?
    for _, thread in data.iteritems():
        if skip_no_answer:
            # No answer given
            if len(thread) == 1:
                continue

        thread_vocab = set()
        for t in thread:
            thread_vocab.update(extract_text_vocab(t))

        vocab.update(thread_vocab)

    return vocab


def gen_vocab_file():
    """
    Provide a list of data files (in this case of json-encoded SO and mailman)
    and process to create a vocab file.
    :param data_files:
    :return:
    """
    total_vocab = set()
    so_vocab = get_so_vocab("snlp_so_questions.json")
    mailman_vocab = get_mailman_vocab("nlp_user_questions_space.json")

    total_vocab.update(so_vocab)
    total_vocab.update(mailman_vocab)

    print "Len so vocab: ", len(so_vocab)
    print "Len mailman vocab: ", len(mailman_vocab)
    print "Len total vocab: ", len(total_vocab)

    # Note manually add 0 -> EOS, 1 -> <unk> for compatibility with gen_emb.py

    # Generate vocab files for SO
    idx = 2
    with open("so_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")

        for w in so_vocab:
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Generate vocab files for mailman
    idx = 2
    with open("mailman_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")

        for w in mailman_vocab:
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Generate vocab files for combined
    idx = 2
    with open("so+mailman_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")

        for w in total_vocab:
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    return so_vocab, mailman_vocab, total_vocab


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
    so, mailman, combined = gen_vocab_file()
    glv_vocab = get_glv_vocab("/Users/mihaileric/Documents/Research/LSTM-NLI/data/glove.6B.50d.txt.gz")

    print "Len missing SO: ", len(so.difference(glv_vocab))
    print "Len missing mailman: ", len(mailman.difference(glv_vocab))
    print "Len missing combined: ", len(combined.difference(glv_vocab))