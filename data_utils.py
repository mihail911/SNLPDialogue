import collections
import json


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
    #so_data_statistics("questions.json")
    lists_data_statistics("nlp_user_questions_space.json")
