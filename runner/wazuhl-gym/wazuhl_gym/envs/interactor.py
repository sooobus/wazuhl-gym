from pymongo import MongoClient, DESCENDING
import numpy as np
import random
import time
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
        self.intended_cases = self.db["intended_cases"]
        self.compiled_cases = self.db["compiled_cases"]
        self.next_index = 0

    def _get_state(self, index):
        record = self.states.find_one({"index": index})
        while not record:
            record = self.states.find_one({"index": index})
        index, state, done = record["index"], record["state"], record["done"]
        #if done:
        #    ts = time.time()
        #    new_ts = time.time()
        #    compiled_record = None
        #    while not compiled_record and new_ts - ts < 30: #TODO: what is correct waiting time? Must be less
        #        compiled_record = self.compiled_cases.find_one({"index": index})
        #        new_ts = time.time()
        #    if not compiled_record:
        #        done = False
        #    else:
        #        self.compiled_cases.delete_one({"index": index})
        return index, np.array(state), done

    def get_state(self):
        return self._get_state(self.next_index - 1)

    def send_action(self, name):
        self.actions.insert_one({"index": self.next_index, "action": name})
        logging.info("Sent action " + name)
        self.next_index += 1

    def _get_rewards(self, i):
        logging.debug("All runned indices:")
        logging.debug([record["index"] for record in self.test_results.find()])
        record = self.test_results.find_one({"index": i})
        ts = time.time()
        new_ts = time.time()
        while not record and new_ts - ts < 120:
            record = self.test_results.find_one({"index": i})
            new_ts = time.time()
        if not record:
            return None, None
        compile_time, exec_time = record["compile_time"], record["exec_time"]
        return compile_time, exec_time

    def get_rewards(self): #can be called only once per done
        record = self.intended_cases.find_one(sort=[("index", DESCENDING)])
        logging.debug("Current case: {}".format(record["index"]))
        assert record
        return self._get_rewards(record["index"])

    def get_possible_actions(self):
        record = self.possible_actions.find_one({})
        while not record:
            record = self.possible_actions.find_one({})
        actions = record["actions"]
        return actions