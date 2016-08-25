import collections
import dill as pickle
import sqlite3
import json
import os
import os.path
import pprint

from create_data import tokenize_data, gen_data_split
from data_utils import extract_text_vocab, compute_data_len


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
        print "Creating DB"
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


attr_names = ['name', 'R_post_code', 'R_cuisine', 'R_location', 'R_phone', 'R_address',
                  'R_price', 'R_rating']

def format_attr(restr_list):
    """
    Return list of tuples of restaurant info formatted as:
    (name, post_code, cuisine, location, phone, address, price, rating)
    :param restr_attr:  dict of restaurant
    :return:
    """
    restr_tuples = []
    for rest in restr_list:
        attr = []
        for n in attr_names:
            try:
                attr.append(restr_list[rest][n])
            except:
                attr.append('')

        restr_tuples.append(tuple(attr))

    return restr_tuples


def process_api_results(api_results):
    """
    Process api results extracting restaurant information
    and return tuples of restaurant info
    :param api_results:
    :return:
    """

    restaurant_info = collections.defaultdict(dict)
    for idx, result in enumerate(api_results):
        values = result.split(" ")

        # Populate dict of restaurant
        restaurant_info[values[0]]['name'] = values[0]
        restaurant_info[values[0]][values[1]] = values[2]


    restaurants = format_attr(restaurant_info)

    return restaurants


def get_dialogue_restr(dialogue_file, db):
    """
    Save dict mapping from dialogue number to set of potential candidates
    :param dialogue_file:
    :param db:
    :return:
    """
    c = sqlite3.connect(db)
    curs = c.cursor()

    with open(dialogue_file, "r") as f:
        dialogues = pickle.load(f)

    for idx, dial in enumerate(dialogues):
        dial = dial[::-1]
        for _, system in dial:
            tokens = system.split()
            api_call = []
            # Found an api_call
            if tokens[0] == "api_call":
                for t in tokens[1:]:
                    if t in attr_names:
                        api_call.append("%")
                    else:
                        api_call.append(t)

                api_call = tuple(api_call)
                curs.execute("SELECT * FROM Restaurants WHERE cuisine LIKE ? "
                         "and location LIKE ? and price LIKE ?", api_call)
                api_response = curs.fetchall()
                rests = set([entry[0] for entry in api_response])
                
                break


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
    word_to_idx = {}
    vocab_set = set()

    f_dialogue = open(dialogue_file, "r")
    dialogues = pickle.load(f_dialogue)
    count = 0
    for dialogue in dialogues:
        for user, system in dialogue:
            user_set, user_tokens = extract_text_vocab(user, re_patterns)
            system_set, system_tokens = extract_text_vocab(system, re_patterns)

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

    f_dialogue = open(filename, "r")
    dialogues = pickle.load(f_dialogue)

    outfile = open(outfilename, "w")


    for idx, dialogue in enumerate(dialogues):
        curr_src = ""

        for user, system in dialogue:
            src = curr_src + " " + user
            target = system
            outfile.write(str(idx) + "\t" + target + "\t" + src + "\n")

            # Update curr_src
            curr_src += " " + user + " " + system


    f_dialogue.close()
    outfile.close()



train_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt"
dev_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-dev.txt"
test_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst.txt"

train_pickle = "train_dialogues.pkl"
dev_pickle = "dev_dialogues.pkl"
test_pickle = "test_dialogues.pkl"
all_pickle = "/Users/mihaileric/Documents/Research/Ford Project/textsum/src/data/dstc2_all_dialogues.pkl"

db_file = "/Users/mihaileric/Documents/Research/SNLPDialogue/data/dstc2.db"


#extract_dialogues(train_filename, train_pickle, restaurant_db=db_file)
#extract_dialogues(dev_filename, dev_pickle, restaurant_db=db_file)
#extract_dialogues(test_filename, test_pickle, restaurant_db=db_file)
#
# # Consolidate
# consolidate_dialogues(train_pickle, dev_pickle, test_pickle, all_pickle)

get_dialogue_restr("dstc2_all_dialogues.pkl", "dstc2.db")


# word_to_idx = extract_dialogue_vocab(all_pickle, db_file, "dstc2_vocab.txt")
# create_dialogues_file(all_pickle, "dstc2_sentences.txt")
# tokenize_data("dstc2_sentences.txt", "dstc2_tok.txt", "dstc2_par_sent.txt",
#               word_to_idx, re_patterns)
#
# gen_data_split("/Users/mihaileric/Documents/Research/SNLPDialogue/data/", "dstc2", [0.8, 0.1, 0.1])

# compute_data_len("dstc2_sentences.txt")

