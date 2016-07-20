import dill as pickle
import sqlite3
import json
import os
import os.path

from create_data import tokenize_data, gen_data_split
from data_utils import extract_text_vocab


def extract_dialogues(filename, pkl_filename, restaurant_db):
    """
    Extract dialogues from given filename as list of lists
    :param filename:
    :return:
    """
    dialogues = []

    # Create DB
    if not os.path.exists(restaurant_db):
        conn = sqlite3.connect(restaurant_db)
        c = conn.cursor()
        c.execute("""CREATE TABLE Restaurants (name text unique, post_code text, cuisine text, location text,
              phone text, address text, price text, rating text)""")
        conn.commit()
    else:
        conn = sqlite3.connect(restaurant_db)
        c = conn.cursor()


    with open(filename, "r") as f:
        exchanges = []
        # (Post_code, cuisine, location, phone, address, price, rating)
        api_results = []

        for line in f:
            # Signifies that end of dialogue has been reached so
            # output utterances
            if line == "\n":
                dialogues.append(exchanges)
                restaurants = process_api_results(api_results)

                # Update restaurants in DB
                if len(restaurants) != 0:
                    for r in restaurants:
                        c.execute("INSERT OR IGNORE INTO Restaurants VALUES "
                                  "(?,?,?,?,?,?,?,?)", r)
                        conn.commit()

                exchanges = []
                api_results = []
                continue

            contents = line.strip().split("\t")
            if len(contents) == 1:
                clean_contents = " ".join(contents[0].strip().split(" ")[1:])
                if clean_contents != "" and clean_contents != "api_call no result":
                    api_results.append(clean_contents)

            else:
                user, system = contents[0], contents[1]
                user = " ".join(user.split(" ")[1:])

                exchanges.append((user, system))


    print "Dialogues: ", len(dialogues)
    with open(pkl_filename, "wb") as f:
        pickle.dump(dialogues, f)


def process_api_results(api_results):
    """
    Process api results extracting restaurant information
    and return tuples of restaurant info
    :param api_results:
    :return:
    """
    restaurants = []
    curr_rest = []
    for idx, result in enumerate(api_results):
        values = result.split(" ")
        if len(curr_rest) == 0:
            curr_rest.append(values[0])
        curr_rest.append(values[2])
        if (idx+1) % 7 == 0:
            restaurants.append(tuple(curr_rest))
            curr_rest = []

    return restaurants


def consolidate_dialogues(train_pickle, dev_pickle, test_pickle, outfile):
    """
    Consolidate the pickled for train/dev/test into one so we can
    split later according to our needs.
    :param train_pickle:
    :param dev_pickle:
    :param test_pickle:
    :return:
    """
    f_train = open(train_pickle, "r")
    f_dev = open(dev_pickle, "r")
    f_test = open(test_pickle, "r")

    total_dialogues = []
    total_dialogues.extend(pickle.load(f_train))
    total_dialogues.extend(pickle.load(f_dev))
    total_dialogues.extend(pickle.load(f_test))

    f_train.close()
    f_dev.close()
    f_test.close()

    with open(outfile, "wb") as f:
        pickle.dump(total_dialogues, f)



re_patterns = r"<|>|[\w]+|,|\?|\.|\(|\)|\\|\"|\/|;|\#|\&|\$|\%|\@|\{|\}|\+|\-|\:"

def extract_dialogue_vocab(dialogue_file, dialogue_db, outfile_name):
    """
    Extract vocab file and populate word_to_idx mapping
    :param dialogue_file:
    :param dialogue_db:
    :return:
    """
    #re_patterns = r"<|>|[\w]+|,|\?|\.|\(|\)|\\|\"|\/|;|\#|\&|\$|\%|\@|\{|\}|\+|\-|\:"
    word_to_idx = {}
    vocab_set = set()

    f_dialogue = open(dialogue_file, "r")
    dialogues = pickle.load(f_dialogue)
    count = 0
    for dialogue in dialogues:
        for user, system in dialogue:
            user_set, user_tokens = extract_text_vocab(user, re_patterns)
            system_set, system_tokens = extract_text_vocab(system, re_patterns)

            # print user
            # print system
            # print user_tokens
            # print system_tokens
            # print "-"*100
            count += 1

            vocab_set.update(system_set)
            vocab_set.update(user_set)

    f_dialogue.close()

    # Also get vocab from database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT * FROM Restaurants")
    entries = c.fetchall()
    for e in entries:
        vocab_set.update(set(e))

    #print vocab_set, len(vocab_set)

    # Output vocab mapping to file
    idx = 2
    with open(outfile_name, "wb") as f:
        f.write("0" + "\t" + "eos" + "\n")
        f.write("1" + "\t" + "<unk>" + "\n")
        word_to_idx["eos"] = 0
        word_to_idx["<unk>"] = 1

        for w in vocab_set:
            if w == "eos": continue
            word_to_idx[w] = idx
            f.write(str(idx) + "\t" + w + "\n")
            idx += 1


    return word_to_idx


# NOTE: Data file is outputted as target -> src for consistency with seq2seq reader implementation
def create_dialogues_file(filename, outfilename):
    """
    Generate filename for dialogues
    :param filename:
    :return:
    """
    d_idx = 1

    f_dialogue = open(filename, "r")
    dialogues = pickle.load(f_dialogue)

    outfile = open(outfilename, "w")


    for dialogue in dialogues:
        curr_src = ""
        for user, system in dialogue:
            src = curr_src + " " + user
            target = system
            outfile.write(str(d_idx) + "\t" + target + "\t" + src + "\n")

            # Update curr_src
            curr_src += " " + user + " " + system

            d_idx += 1

    f_dialogue.close()
    outfile.close()

# def create_tokenized_data(sentences_file, word_to_idx, tok_outfile, p_se):
#     """
#     Create tokenized data file
#     :param sentences_file:
#     :return:
#     """


train_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt"
dev_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-dev.txt"
test_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst.txt"

train_pickle = "train_dialogues.pkl"
dev_pickle = "dev_dialogues.pkl"
test_pickle = "test_dialogues.pkl"
all_pickle = "/Users/mihaileric/Documents/Research/Ford Project/textsum/src/data/dstc2_all_dialogues.pkl"

db_file = "/Users/mihaileric/Documents/Research/Ford Project/textsum/src/data/dstc2.db"


# extract_dialogues(train_filename, train_pickle, restaurant_db=db_file)
# extract_dialogues(dev_filename, dev_pickle, restaurant_db=db_file)
# extract_dialogues(test_filename, test_pickle, restaurant_db=db_file)
#
# # Consolidate
# consolidate_dialogues(train_pickle, dev_pickle, test_pickle, all_pickle)



word_to_idx = extract_dialogue_vocab(all_pickle, db_file, "dstc2_vocab.txt")
#create_dialogues_file(all_pickle, "dstc2_sentences.txt")
#tokenize_data("dstc2_sentences.txt", "dstc2_tok.txt", "dstc2_par_sent.txt",
#              word_to_idx, re_patterns)

gen_data_split("/Users/mihaileric/Documents/Research/SNLPDialogue/data/", "dstc2", [0.8, 0.1, 0.1])