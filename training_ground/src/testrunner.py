import logging
import random
from tqdm import tqdm

from src import config


def get_tests(suites, flag):
    tests = []
    for suite in suites:
        logging.info("Suite: {}".format(suite))
        option = '-C/wazuhl/training_ground/cmake_configs/{}.cmake'
        suite.configure(config.get_clang(), config.get_clangpp(), option.format(flag))
        logging.info("Configured suite and preparing to get tests from it")
        tests.extend(suite.get_tests())
        logging.info(len(suite.get_tests()))
        logging.info(len(tests))
    return tests


def run(tests):
    for test in tqdm(tests):
        run_test(test)
    return tests


def run_random(tests):
    assert len(tests) > 0, "Please provide some tests to run_random"
    logging.debug("Running some random test")
    test = random.choice(tests)
    logging.debug("Chosen a test")
    return run_test(test)


def run_test(test, interactor=None, index=0):
    logging.debug("Compile test %s", test)
    test.compile()
    if interactor:
        interactor.send_compiled(index, test)
    logging.debug("Run test %s", test)
    test.run()
    #logging.debug("Cleanup test %s", test)
    #test.clean()
    return test
