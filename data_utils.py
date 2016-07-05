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


def gen_data(so_data_fn, mailman_data_fn):
    """
    Output data to desired format (i.e. ex. id \t src utterance \t tgt utterance).
    SO Data will output dialogues for the following sequences: Q -> [A_1, ..., A_k],
    Q -> [C_1, ..., C_k], and A -> [C_1, ..., C_k] where A_j and C_j denote jth and answer
    and jth comment in sequence for a given question (Q) or answer

    :param so_data_fn Filename containing Stack overflow data (None if not using this data)
    :param mailman_data_fn Filename containing mailman data (None if not using)
    :return:
    """
    output_file = open("data/data_sentences.txt", "a+")

    a_idx = 1
    if so_data_fn:
        with open(so_data_fn, "rb") as f:
            so_data = json.load(f)
    else:
        so_data = None

    if mailman_data_fn:
        with open(mailman_data_fn, "rb") as f:
            mailman_data = json.load(f)
    else:
        mailman_data = None

    # Read in so_data and output to file
    if so_data:
        for question in so_data:
            # TODO: Whether to include text of question title?
            question = json.loads(question)

            q_body = question["body"].encode("utf-8")
            q_body = clean_so_text(q_body)

            answers = question["answers"]
            comments = question["comments"]

            # There is no dialogue because no comments/answers to question
            if len(answers) == 0 and len(comments) == 0:
                continue

            # Create dialogue of form (Q, C_1), (Q+C_1, C_2), etc.
            curr_c = ""
            for c in comments:
                c = c.encode("utf-8")
                c = clean_so_text(c)
                src = q_body + curr_c
                target = c
                output_file.write(str(a_idx) + "\t" + src + "\t" + "EOS" + " " + target + " " + "EOS" + "\n")

                curr_c += " " + c

                a_idx += 1

            # Create dialogue of form (Q, A_1), (Q+A_1, A_2), etc.
            curr_a = ""
            for a in answers:
                a_text = a["text"].encode("utf-8")
                a_text = clean_so_text(a_text)
                src = q_body + curr_a
                target = a_text
                output_file.write(str(a_idx) + "\t" + src + "\t" + "EOS" + " " + target + " " + "EOS" + "\n")

                curr_a += " " + a_text
                a_idx += 1

                # Also of form (A_1, C_11), (A_1+C_11, C_21), etc.
                a_comments = a["comments"]
                curr_a_c = ""
                for a_c in a_comments:
                    a_c = a_c.encode("utf-8")
                    a_c = clean_so_text(a_c)
                    src = a_text + curr_a_c
                    target = a_c

                    output_file.write(str(a_idx) + "\t" + src + "\t" + "EOS" + " " + target + " " + "EOS" + "\n")

                    curr_a_c += " " + a_c
                    a_idx += 1

    # Read in mailman_data and output to file
    if mailman_data:
        for _, thread in mailman_data.iteritems():
            # No answer given so no valid dialogue
            if len(thread) == 1:
                continue

            question = thread[0].encode("utf-8)")
            curr_a = ""
            for t in thread:
                # TODO: Remove "-----" string
                t = t.encode("utf-8")
                src = question + curr_a
                target = t
                output_file.write(str(a_idx) + "\t" + src + "\t" + "EOS" + " " + target + " " + "EOS" + "\n")

                curr_a += " " + t
                a_idx += 1

    output_file.close()


def tokenize_data(data_file, vocab_word_to_idx):
    """
    Convert data files from word tokens to idx tokens given data word_to_idx file
    for vocab mapping
    :param data_file:
    :param vocab_word_to_idx:
    :return:
    """
    with open(data_file, "rb") as f:
        tokenized_file = open("data/data_tokenized.txt", "wb")

        idx = 1
        count = 0
        for example in f:
            if count > 5: break

            idx, src, target = example.split("\t")
            _, src_tokens = extract_text_vocab(src)
            _, target_tokens = extract_text_vocab(target)

            tokenized_file.write(str(idx) + "\t")

            # Write src tokens indices
            for s in src_tokens:
                tokenized_file.write(str(vocab_word_to_idx[s]) + " ")

            tokenized_file.write("\t")
            # Write target tokens indices
            for t in target_tokens:
                tokenized_file.write(str(vocab_word_to_idx[t]) + " ")

            tokenized_file.write("\n")

            count += 1
            idx += 1

        tokenized_file.close()


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
    Tokenize text and return a set and list of vocab words
    :param text:
    :return:
    """
    text_tokens = re.findall(r"<|>|[\w]+|,|\?|\.|\(|\)|\\|\"|\/|;|\#|\&|\$|\%|\@|\{|\}|\+|\-|\:", text)
    lower_tokens = [t.lower() for t in text_tokens]
    return set(lower_tokens), lower_tokens


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
    vocab_freq = collections.Counter()

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
        body_set, body_list = extract_text_vocab(q_body)
        vocab_freq.update(body_set)
        vocab.update(body_set)

        # Extract vocab from question comments
        for c in comments:
            c = c.encode("utf-8")
            c = clean_so_text(c)
            c_voc, c_list = extract_text_vocab(c)
            vocab_freq.update(c_voc)
            vocab.update(c_voc)

        # Extract vocab from question answers and answer comments
        for a in answers:
            a_text = a["text"].encode("utf-8")
            a_text = clean_so_text(a_text)
            a_voc, a_list = extract_text_vocab(a_text)
            vocab_freq.update(a_voc)
            vocab.update(a_voc)

            a_comments = a["comments"]
            for a_c in a_comments:
                a_c = a_c.encode("utf-8")
                a_c = clean_so_text(a_c)
                a_c_vocab, a_c_list = extract_text_vocab(a_c)
                vocab_freq.update(a_c_vocab)
                vocab.update(a_c_vocab)


    return vocab, vocab_freq


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
    vocab_freq = collections.Counter()

    # TODO: Whether to process title for vocab?
    for _, thread in data.iteritems():
        if skip_no_answer:
            # No answer given
            if len(thread) == 1:
                continue

        thread_vocab = set()
        for t in thread:
            thread_voc, thread_list = extract_text_vocab(t)
            vocab_freq.update(thread_voc)
            thread_vocab.update(thread_voc)

        vocab.update(thread_vocab)


    return vocab, vocab_freq


def gen_vocab_file():
    """
    Provide a list of data files (in this case of json-encoded SO and mailman)
    and process to create a vocab file.
    :param data_files:
    :return:
    """
    total_word_to_idx = {}
    total_vocab = set()
    total_freq = collections.Counter()
    so_vocab, so_freq = get_so_vocab("data/snlp_so_questions.json")
    mailman_vocab, mailman_freq = get_mailman_vocab("data/nlp_user_questions_space.json")

    # Update total vocab set
    total_vocab.update(so_vocab)
    total_vocab.update(mailman_vocab)

    # Update total freq set
    total_freq.update(so_freq)
    total_freq.update(mailman_freq)

    print "Len so vocab: ", len(so_vocab)
    print "Len mailman vocab: ", len(mailman_vocab)
    print "Len total vocab: ", len(total_vocab)

    # Note manually add 0 -> EOS, 1 -> <unk> for compatibility with gen_emb.py

    # Generate vocab files for SO
    idx = 2
    so_word_to_idx = {}
    with open("data/so_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")
        so_word_to_idx["EOS"] = 0
        so_word_to_idx["<unk>"] = 1

        for w in so_vocab:
            so_word_to_idx[w] = idx
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Generate vocab files for mailman
    idx = 2
    mailman_word_to_idx = {}
    with open("data/mailman_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")
        mailman_word_to_idx["EOS"] = 0
        mailman_word_to_idx["<unk>"] = 1

        for w in mailman_vocab:
            mailman_word_to_idx[w] = idx
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Generate vocab files for combined
    idx = 2
    total_word_to_idx = {}
    with open("data/so+mailman_vocab.txt", "wb") as f:
        f.write("0" + "\t" + "EOS" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")
        total_word_to_idx["EOS"] = 0
        total_word_to_idx["<unk>"] = 1

        for w in total_vocab:
            total_word_to_idx[w] = idx
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Pickle mapping files
    with open("data/so_word_to_idx.pkl", "wb") as f:
        pickle.dump(so_word_to_idx, f)

    with open("data/mailman_word_to_idx.pkl", "wb") as f:
        pickle.dump(mailman_word_to_idx, f)

    with open("data/total_word_to_idx.pkl", "wb") as f:
        pickle.dump(total_word_to_idx, f)

    return so_vocab, mailman_vocab, total_vocab,\
           so_word_to_idx, mailman_word_to_idx, total_word_to_idx


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