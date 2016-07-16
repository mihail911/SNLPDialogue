import dill as pickle
import sqlite3
import json
import os
import os.path


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


train_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt"
dev_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-dev.txt"
test_filename = "/Users/mihaileric/Documents/Research/Data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst.txt"

train_pickle = "train_dialogues.pkl"
dev_pickle = "dev_dialogues.pkl"
test_pickle = "test_dialogues.pkl"
all_pickle = "dstc2_all_dialogues.pkl"

db_file = "dstc2.db"


extract_dialogues(train_filename, train_pickle, restaurant_db=db_file)
extract_dialogues(dev_filename, dev_pickle, restaurant_db=db_file)
extract_dialogues(test_filename, test_pickle, restaurant_db=db_file)

# Consolidate
consolidate_dialogues(train_pickle, dev_pickle, test_pickle, all_pickle)
