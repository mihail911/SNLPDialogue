import argparse
import json
import numpy as np

from data_utils import clean_text, extract_text_vocab
from vocab import gen_vocab_file


def gen_data_split(data_file, prefix, split):
    """
    Provide a tuple train/dev/test split for creating appropriate data splits
    :param data_file: Tokenized data file
    :param data_parallel_sent: Corresponding data file with actual text
    :return:
    """
    total_points = 0.0
    with open(data_file + prefix + "_tok.txt", "r") as f:
        for line in f:
            total_points += 1

    random_idx = np.random.permutation(np.arange(total_points))

    first_split = split[0] * total_points
    second_split = round(first_split + split[1] * total_points)
    idx_splits = np.split(random_idx, [first_split, second_split])

    train, val, test = set(idx_splits[0]), set(idx_splits[1]), set(idx_splits[2])

    train_file = open(data_file + prefix + "_train_tok.txt", "w")
    val_file = open(data_file + prefix + "_val_tok.txt", "w")
    test_file = open(data_file + prefix + "_test_tok.txt", "w")

    train_sent_file = open(data_file + prefix + "_train_sent.txt", "w")
    val_sent_file = open(data_file + prefix + "_val_sent.txt", "w")
    test_sent_file = open(data_file + prefix + "_test_sent.txt", "w")

    idx = 0.0
    f_sent = open(data_file + prefix + "_par_sent.txt", "r")
    with open(data_file + prefix + "_tok.txt", "r") as f:

        for tok_point, sent_point in zip(f, f_sent):
           if idx in train:
               train_file.write(tok_point)
               train_sent_file.write(sent_point)
           elif idx in val:
               val_file.write(tok_point)
               val_sent_file.write(sent_point)
           else:
               test_file.write(tok_point)
               test_sent_file.write(sent_point)

           idx += 1

    f_sent.close()

    train_file.close()
    val_file.close()
    test_file.close()


def gen_data(so_data_fn, mailman_data_fn, sent_outfile):
    """
    Output data to desired format (i.e. ex. id \t src utterance \t tgt utterance).
    SO Data will output dialogues for the following sequences: Q -> [A_1, ..., A_k],
    Q -> [C_1, ..., C_k], and A -> [C_1, ..., C_k] where A_j and C_j denote jth and answer
    and jth comment in sequence for a given question (Q) or answer

    :param so_data_fn Filename containing Stack overflow data (None if not using this data)
    :param mailman_data_fn Filename containing mailman data (None if not using)
    :return:
    """
    output_file = open(sent_outfile, "w")

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
            q_body = clean_text(q_body)

            answers = question["answers"]
            comments = question["comments"]

            # There is no dialogue because no comments/answers to question
            if len(answers) == 0 and len(comments) == 0:
                continue

            # Create dialogue of form (Q, C_1), (Q+C_1, C_2), etc.
            curr_c = ""
            for c in comments:
                c = c.encode("utf-8")
                c = clean_text(c)
                src = q_body + curr_c
                target = c
                output_file.write(str(a_idx) + "\t" + target + "\t" + src + "\n")

                curr_c += " " + c

                a_idx += 1

            # Create dialogue of form (Q, A_1), (Q+A_1, A_2), etc.
            curr_a = ""
            for a in answers:
                a_text = a["text"].encode("utf-8")
                a_text = clean_text(a_text)
                src = q_body + curr_a
                target = a_text
                output_file.write(str(a_idx) + "\t" + target + "\t" + src + "\n")

                curr_a += " " + a_text
                a_idx += 1

                # Also of form (A_1, C_11), (A_1+C_11, C_21), etc.
                a_comments = a["comments"]
                curr_a_c = ""
                for a_c in a_comments:
                    a_c = a_c.encode("utf-8")
                    a_c = clean_text(a_c)
                    src = a_text + curr_a_c
                    target = a_c

                    output_file.write(str(a_idx) + "\t" + target + "\t" + src + "\n")

                    curr_a_c += " " + a_c
                    a_idx += 1

    # Read in mailman_data and output to file
    if mailman_data:
        for _, thread in mailman_data.iteritems():
            # No answer given so no valid dialogue
            if len(thread) == 1:
                continue

            question = thread[0].encode("utf-8)")
            question = clean_text(question)
            curr_a = ""
            for t in thread[1:]:
                # TODO: Remove "-----" string
                t = t.encode("utf-8")
                t = clean_text(t)
                src = question + curr_a
                target = t
                if t == "":
                    continue

                output_file.write(str(a_idx) + "\t" + target + "\t" + src + "\n")

                curr_a += " " + t
                a_idx += 1

    output_file.close()


def tokenize_data(data_file, tok_outfile, p_sent_file, vocab_word_to_idx, re_patterns):
    """
    Convert data files from word tokens to idx tokens given data word_to_idx file
    for vocab mapping
    :param data_file:
    :param tok_outfile: output file for tokens of data
    :param p_sent_file: output file with text of data corresponding to tokenized version
    :param vocab_word_to_idx:
    :return:
    """
    with open(data_file, "rb") as f:
        tokenized_file = open(tok_outfile, "wb")
        parallel_sent_file = open(p_sent_file, "wb")

        for example in f:
            idx, src, target = example.split("\t")
            _, src_tokens = extract_text_vocab(src, re_patterns)
            _, target_tokens = extract_text_vocab(target, re_patterns)

            tokenized_file.write(str(idx) + "\t")
            parallel_sent_file.write(str(idx) + "\t")

            # Write target tokens indices
            for t in target_tokens:
                tokenized_file.write(str(vocab_word_to_idx[t]) + " ")
                parallel_sent_file.write(str(t) + " ")

            tokenized_file.write("\t")
            parallel_sent_file.write("\t")

            # Write source tokens indices
            for s in src_tokens:
                tokenized_file.write(str(vocab_word_to_idx[s]) + " ")
                parallel_sent_file.write(str(s) + " ")

            tokenized_file.write("\n")
            parallel_sent_file.write("\n")

        tokenized_file.close()
        parallel_sent_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for data generation")
    parser.add_argument("--data_dir", type=str, help="directory containing all data files")
    args = vars(parser.parse_args())

    data_dir = args["data_dir"]
    # sent_outfile = "/Users/mihaileric/Documents/Research/Ford Project/textsum/src/data/"
    # tokenized_outfile = "/Users/mihaileric/Documents/Research/Ford Project/textsum/src/data/"

    _, _, _, so_word_to_idx, mailman_word_to_idx, total_word_to_idx = gen_vocab_file(data_dir)
    gen_data("data/snlp_so_questions.json", "data/nlp_user_questions_space.json", data_dir + "data_sentences.txt")
    tokenize_data(data_dir + "data_sentences.txt", data_dir + "data_tokenized.txt",
                  data_dir + "data_parallel_sentences.txt", total_word_to_idx)

    print "Generating data split..."
    split = [0.8, 0.1, 0.1]
    gen_data_split(data_dir, split)