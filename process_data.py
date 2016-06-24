import cPickle as pickle
import json
import stackexchange
import sys
import time

class Question(object):
    """
    Custom type representing question with desired information
    """
    def __init__(self, id, link, title, body, comments, answers):
        """
        :param id: question id
        :param link: question link on stack overflow
        :param title: question title
        :param body: question body -- note this is unprocessed HTML, tags and all
        :param comments: list of all comments on question (might be empty, naturally)
        :param answers: list of form {text: answer_text, comments: [list of comment bodies]}
        """
        self.link = link
        self.title = title
        self.id = id
        self.body = body
        self.comments = comments
        self.answers = answers


    def to_json(self):
        """
        Output question in desired JSON format for writing to disk:
        {question: title, body: question_text, answers: [list of form {text: answer_text, comments: [list of comment_text]}], comments: [list of comments_text]}
        """
        q_data = {"question": self.title, "body": self.body, "comments": self.comments, "answers": self.answers}
        return json.dumps(q_data)


class SNLPData(object):
    """
    Object containing all info related to StackOverflow questions tagged
    with "stanford-nlp"
    """
    def __init__(self):
        pass


    def pickle_dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.questions, f)


    def json_dump(self, filename):
        """
        Dump the data of all the questions to disk in json-encoded format
        :return:
        """
        questions_in_json = []
        for q in self.questions:
            questions_in_json.append(q.to_json())

        with open(filename, "wb") as f:
            json.dump(questions_in_json, f)

        return questions_in_json


    def question_data(self, tags="stanford-nlp"):
        """
        Processes all questions for given tag in SO and extracts id, link, and title
        :param tags:
        :return:
        """
        nlp_qlink = []
        nlp_qid = []
        nlp_qtitle = []
        snlp_questions = []

        so = stackexchange.Site(stackexchange.StackOverflow, "r*tSlw7sx8F7mBqKB6RGiA((")

        # To ensure that body and comments are also included
        so.be_inclusive()
        all_questions = so.questions
        count = 0
        start = time.time()
        for q in all_questions(tagged=tags):
            print "Question number: ", count
            nlp_qid.append(q.question_id)
            nlp_qlink.append(q.link)
            nlp_qtitle.append(q.title)

            # Get question comments body if any
            q_comments = []
            try:
                q.comments.fetch()
            except:
                print "Error encountered when fetching question comment. Skipping question..."
                continue

            for q_c in q.comments:
                q_comments.append(q_c.body)

            # Note reversing the list to preserve the order of comments in the original post
            q_comments = q_comments[::-1]

            # Note question body is unfiltered with HTML tags still in place
            answers = []
            for a in q.answers:
                # Process all comments
                try:
                    a.comments.fetch()
                except:
                    print "Error encountered when fetching answer comment. Skipping answer..."
                    continue

                answer_comments = []
                for c in a.comments:
                    answer_comments.append(c.body)
                # Note the reversing of the comments list to preserve the order in the original post
                answers.append({"text": a.body, "comments": answer_comments[::-1]})

            question = Question(q.id, q.link, q.title, q.body, q_comments, answers)
            snlp_questions.append(question)

            count += 1

        print "Finished processing questions and it took {0} seconds".format(str(time.time() - start))

        self.questions = snlp_questions
        return snlp_questions


if __name__ == "__main__":
    data = SNLPData()
