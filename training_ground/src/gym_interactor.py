from pymongo import MongoClient, DESCENDING
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Interactor:
    def __init__(self):
        self.client = MongoClient('mongo')
        self.db = self.client["wazuhl"]
        self.test_results = self.db["test_results"]
        self.intended_cases = self.db["intended_cases"]
        self.compiled_cases = self.db["compiled_cases"]

    def send_test_result(self, index, test):
        self.test_results.insert_one({"index": index, "test_name": str(test),
                                      "compile_time": test.compile_time,
                                      "exec_time": test.execution_time})

    def send_intended_test_id(self, index, test):
        self.intended_cases.insert_one({"index": index, "test_name": str(test)})

    def send_compiled(self, index, test):
        self.compiled_cases.insert_one({"index": index, "test_name": str(test)})