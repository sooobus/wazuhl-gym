from pymongo import MongoClient, DESCENDING
import numpy as np
import random
import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Interactor:
    def __init__(self):
        self.client = MongoClient('mongo')
        self.db = self.client["wazuhl"]
        self.actions = self.db["actions"]
        self.possible_actions = self.db["possible_actions"]
        self.states = self.db["states"]
        self.test_results = self.db["test_results"]
        self.next_index = 0
        self.next_test_index = 0

    def _get_state(self, index):
        record = self.states.find_one({"index": index})
        while not record:
            record = self.states.find_one({"index": index})
        index, state, done = record["index"], record["state"], record["done"]
        return index, np.array(state), done

    def get_state(self):
        return self._get_state(self.next_index - 1)

    def send_action(self, name):
        self.actions.insert_one({"index": self.next_index, "action": name})
        logging.info("Sent action " + name)
        self.next_index += 1

    def _get_rewards(self, i):
        record = self.test_results.find_one({"index": i})
        while not record:
            record = self.test_results.find_one({"index": i})
        compile_time, exec_time = record["compile_time"], record["exec_time"]
        self.next_test_index += 1
        return compile_time, exec_time

    def get_rewards(self): #can be called only once per done
        return self._get_rewards(self.next_test_index)

    def get_possible_actions(self):
        record = self.possible_actions.find_one({})
        while not record:
            record = self.possible_actions.find_one({})
        actions = record["actions"]
        return actions