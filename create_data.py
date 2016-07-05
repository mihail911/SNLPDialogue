import json

from data_utils import clean_text, extract_text_vocab
from vocab import gen_vocab_file


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
                output_file.write(str(a_idx) + "\t" + src + "\t" + "EOS" + " " + target + " " + "EOS" + "\n")

                curr_c += " " + c

                a_idx += 1

            # Create dialogue of form (Q, A_1), (Q+A_1, A_2), etc.
            curr_a = ""
            for a in answers:
                a_text = a["text"].encode("utf-8")
                a_text = clean_text(a_text)
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
                    a_c = clean_text(a_c)
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
            question = clean_text(question)
            curr_a = ""
            for t in thread[1:]:
                # TODO: Remove "-----" string
                t = t.encode("utf-8")
                t = clean_text(t)
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

        for example in f:
            idx, src, target = example.split("\t")
            _, src_tokens = extract_text_vocab(src)
            _, target_tokens = extract_text_vocab(target)

            tokenized_file.write(str(idx) + "\t")

            # Write src tokens indices
            for s in src_tokens:
                #print "-" * 100
                #print "Source: ", s
                tokenized_file.write(str(vocab_word_to_idx[s]) + " ")

            tokenized_file.write("\t")

            # Write target tokens indices
            for t in target_tokens:
                tokenized_file.write(str(vocab_word_to_idx[t]) + " ")

            tokenized_file.write("\n")


        tokenized_file.close()


if __name__ == "__main__":
    _, _, _, so_word_to_idx, mailman_word_to_idx, total_word_to_idx = gen_vocab_file()
    gen_data("data/snlp_so_questions.json", "data/nlp_user_questions_space.json")
    tokenize_data("data/data_sentences.txt", total_word_to_idx)