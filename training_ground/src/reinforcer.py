import logging
import pickle
import os
import random

from src import config
from src import testrunner
from src import utils
#from src import experience
from src import gym_interactor


class Reinforcer:
    def __init__(self, suites):
        self.suites = suites
        self.alpha = config.get_alpha()
        self.tests = []
        #self.experience = experience.Experience()
        self.baselines = {"compile_time": {}, "execution_time": {}}
        self.gym_interactor = gym_interactor.Interactor()

    def measure_baseline(self, flags, rerun=False):
        logging.info("Measuring baseline")
        cache_file = Reinforcer.__etalon_file__("baselines", flags)
        logging.info(cache_file)
        self.baselines = self.__load_cache__(cache_file)
        if not self.baselines or rerun:
            logging.info("Measuring baseline from scratch")
            self.baselines = {"compile_time": {}, "execution_time": {}}
            self.tests = testrunner.get_tests(self.suites, flags)
            self.tests = testrunner.run(self.tests)

            for test in self.tests:
                if test.compile_time is not None:
                    self.baselines["compile_time"][test.name] = test.compile_time
                if test.execution_time is not None:
                    self.baselines["execution_time"][test.name] = test.execution_time

            self.__cache__(self.baselines, cache_file)
        logging.info("Baselines measured")
        nones = [p for p in self.baselines["compile_time"]
                 if self.baselines["compile_time"][p] is None]
        logging.info(nones)
        logging.info(self.baselines["compile_time"])

        #del self.baselines["compile_time"]["XSBench"]
        #del self.baselines["execution_time"]["XSBench"]
        #self.tests =
        assert None not in self.baselines["compile_time"].values(), "None in test"

    def calculate_reward(self, test):
        C = self.baselines["compile_time"][test.name]
        E = self.baselines["execution_time"][test.name]
        assert C is not None, test.name in self.baselines["compile_time"]
        alpha = self.alpha
        Cp = test.compile_time
        Ep = test.execution_time
        logging.info("Cp = {}, C = {}, Ep = {}, E = {}".format(Cp, C, Ep, E))
        if not Cp or not Ep: return -100.0
        return 10 + (E - Ep) / (E + 1e-10) + alpha * (C - Cp) / (C + 1e-10)

    def run(self):
        logging.info("Reinforcer running. Getting tests")
        tests = testrunner.get_tests(self.suites, 'OWtrain')
        logging.info("Got tests")
        logging.info("Before check, {} tests".format(len(tests)))
        tests = [test for test in tests
                 if str(test) in self.baselines["compile_time"].keys()]
        self.__check__(tests)
        tests = [test for test in tests
                 if self.__sufficient_test("compile_time", test) and
                 self.__sufficient_test("execution_time", test)]
        logging.info("Checked tests")

        test_number = 0
        while True:
            logging.info("Run random")
            t = random.choice(tests)
            self.gym_interactor.send_intended_test_id(test_number, t)
            result = testrunner.run_test(t, self.gym_interactor, test_number)
            reward = self.calculate_reward(result)
            logging.info("Reward: %s", reward)
            #self.experience.approve(reward)
            self.gym_interactor.send_test_result(test_number, result)
            test_number += 1

    def __sufficient_test(self, category, test):
        baseline = self.baselines[category][str(test)]
        return baseline is not None and baseline > 0.01

    def __check__(self, tests):
        tests = set(map(str, tests))
        logging.info(len(tests))
        compilation_tests = set(self.baselines["compile_time"].keys())
        execution_tests = set(self.baselines["execution_time"].keys())
        message = "Etalon tests ({0}) differ from the ones for Wazuhl!"
        if not tests.issubset(compilation_tests):
            utils.error(message.format("compilation"))
        if not tests.issubset(execution_tests):
            utils.error(message.format("execution"))
        assert len(tests) > 0, "Test suite is empty!"

    def __cache__(self, data, cache_file):
        with open(cache_file, 'wb') as cache:
            pickle.dump(data, cache)

    def __load_cache__(self, cache_file):
        if not os.path.exists(cache_file):
            logging.info("Couldn't find file {0}".format(cache_file))
            return None
        with open(cache_file, 'rb') as cache:
            return pickle.load(cache)

    @staticmethod
    def __etalon_file__(name, flags):
        result = os.path.join(config.get_output(),
                            "{0}.{1}.etalon".format(name, utils.pathify(flags)))
        logging.info("Etalon file for '{0}' lives at '{1}'".
                     format(flags, result))
        return result
