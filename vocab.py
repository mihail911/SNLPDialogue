import collections
import cPickle as pickle
import json

from data_utils import extract_text_vocab, clean_so_text

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