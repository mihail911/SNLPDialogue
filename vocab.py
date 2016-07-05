import collections
import cPickle as pickle
import json

from data_utils import extract_text_vocab, clean_text

"""
General utilities for generating relevant vocabularies from data.
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
        q_body = clean_text(q_body)

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
            c = clean_text(c)
            c_voc, c_list = extract_text_vocab(c)
            vocab_freq.update(c_voc)
            vocab.update(c_voc)

        # Extract vocab from question answers and answer comments
        for a in answers:
            a_text = a["text"].encode("utf-8")
            a_text = clean_text(a_text)
            a_voc, a_list = extract_text_vocab(a_text)
            vocab_freq.update(a_voc)
            vocab.update(a_voc)

            a_comments = a["comments"]
            for a_c in a_comments:
                a_c = a_c.encode("utf-8")
                a_c = clean_text(a_c)
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
            thread_voc, thread_list = extract_text_vocab(clean_text(t))
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
    so_vocab, so_freq = get_so_vocab("data/snlp_so_questions.json", skip_no_answer=True)
    mailman_vocab, mailman_freq = get_mailman_vocab("data/nlp_user_questions_space.json", skip_no_answer=True)

    # Update total vocab set
    total_vocab.update(so_vocab)
    total_vocab.update(mailman_vocab)

    # Update total freq set
    total_freq.update(so_freq)
    total_freq.update(mailman_freq)

    print "Len so vocab: ", len(so_vocab)
    print "Len mailman vocab: ", len(mailman_vocab)
    print "Len total vocab: ", len(total_vocab)

    # Note manually add 0 <-> EOS, 1 <-> <unk> for compatibility with gen_emb.py

    # Generate vocab files for SO
    so_word_to_idx = {}
    write_vocab_file("data/so_vocab.txt", so_vocab, so_word_to_idx)

    # Generate vocab files for mailman
    mailman_word_to_idx = {}
    write_vocab_file("data/mailman_vocab", mailman_vocab, mailman_word_to_idx)

    # Generate vocab files for combined
    total_word_to_idx = {}
    write_vocab_file("data/so+mailman_vocab.txt", total_vocab, total_word_to_idx)

    return so_vocab, mailman_vocab, total_vocab,\
           so_word_to_idx, mailman_word_to_idx, total_word_to_idx


def write_vocab_file(file_name, vocab, word_to_idx):
    """
    Writes vocab file to given file_name and populate word_to_idx mapping
    and pickles mapping
    :param file_name:
    :param word_to_idx:
    :return:
    """
    idx = 2
    with open(file_name + ".txt", "wb") as f:
        f.write("0" + "\t" + "eos" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")
        word_to_idx["eos"] = 0
        word_to_idx["<unk>"] = 1

        for w in vocab:
            if w == "eos": continue
            word_to_idx[w] = idx
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1

    # Pickle resulting word to index mapping
    with open(file_name + ".pkl", "wb") as f:
        pickle.dump(word_to_idx, f)
