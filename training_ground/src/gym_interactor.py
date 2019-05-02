from pymongo import MongoClient, DESCENDING
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Interactor:
    def __init__(self):
        self.client = MongoClient('mongo')
        self.db = self.client["wazuhl"]
        self.test_results = self.db["test_results"]

    def send_test_result(self, index, test):
        self.test_results.insert_one({"index": index, "test_name": str(test),
                                      "compile_time": test.compile_time,
                                      "exec_time": test.execution_time})