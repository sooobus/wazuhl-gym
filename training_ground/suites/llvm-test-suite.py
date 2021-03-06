import logging
import os
import re
import shutil
import subprocess
from src import config, utils


class Suite:
    name = "llvm-test-suite"

    def __init__(self):
        self.tests = []
        self.configuration_env = os.environ.copy()
        self.fake_run = False
        self.build = config.get_llvm_test_suite_build_path()
        self.suite = config.get_llvm_test_suite_path()
        self.caffe = config.get_caffe_bin()

    def get_tests(self):
        return self.tests

    def configure(self, CC, CXX, OPTS):
        logging.info("Run configure with options: CC {}, CXX {}, OPTS {}".format(CC, CXX, OPTS))
        if os.path.exists(self.build):
            shutil.rmtree(self.build)
        os.makedirs(self.build)
        self.go_to_builddir()

        self.configuration_env['CC'] = CC
        self.configuration_env['CXX'] = CXX
        self.configuration_env['LD_LIBRARY_PATH'] = self.caffe

        make_command = ['cmake', self.suite,
                        '-DCMAKE_BUILD_TYPE=Release',
                        OPTS]
        logging.info(make_command)
        with open(os.devnull, 'wb') as devnull:
            cmake_output = subprocess.Popen(make_command, env=self.configuration_env, stdout=subprocess.PIPE)
            out = cmake_output.stdout.read().decode('utf-8')
            logging.debug(out)

        logging.info("Configuration is finished")

        self.__init_tests__()

    def __init_tests__(self):
        utils.check_executable('lit')
        lit = subprocess.Popen(['lit', '--show-tests', self.build], stdout=subprocess.PIPE)
        output = lit.stdout.read().decode('utf-8')
        pattern = r'test-suite :: (.*)'
        results = re.findall(pattern, output)
        self.tests = [Test(os.path.join(self.build, test), self) for test in results]
        logging.info("Len tests (all): {}".format(len(self.tests)))
        self.tests = [test for test in self.tests if "Benchmark" in test.path]
        logging.info("Len tests (benchmark): {}".format(len(self.tests)))

    def go_to_builddir(self):
        os.chdir(self.build)
        logging.debug(self.build + " is a build dir")


class Test:
    def __init__(self, path, suite):
        self.path = path
        self.name = Test.__get_test_name__(path)
        self.suite = suite
        self.compile_time = None
        self.execution_time = None

    @staticmethod
    def __get_test_name__(path):
        _, test_file = os.path.split(path)
        test, _ = os.path.splitext(test_file)
        return test

    def compile(self):
        self.suite.go_to_builddir()
        with open(os.devnull, 'wb') as devnull:
            make_command = ['make', '--always-make', '-j1', self.name]
            logging.debug(make_command)
            #make_output = subprocess.run(make_command,  env=self.suite.configuration_env,
            #                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(make_command,  env=self.suite.configuration_env)
            #logging.debug(make_output)
            #logging.debug("reading output")
            #out = make_output.stdout.decode('utf-8')
            #logging.debug(out)

    def run(self):
        if self.suite.fake_run:
            self.compile_time, self.execution_time = 10000, 10000
            return
        test_run = subprocess.Popen(['lit', self.path], stdout=subprocess.PIPE)
        output = test_run.stdout.read().decode('utf-8')
        logging.debug(output)
        compile_pattern = r'compile_time: (.*)'
        execution_pattern = r'exec_time: (.*)'
        compile_time = re.search(compile_pattern, output)
        execution_time = re.search(execution_pattern, output)
        if compile_time:
            compile_time = float(compile_time.group(1))
        if execution_time:
            execution_time = float(execution_time.group(1))
        self.compile_time, self.execution_time = compile_time, execution_time

    def clean(self):
        self.suite.go_to_builddir()
        #with open(os.devnull, 'wb') as devnull:
        #    make_command = ['make', 'clean']
        #    logging.debug(make_command)
        #    clean_output = subprocess.run(make_command,  env=self.suite.configuration_env,
        #                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    def __str__(self):
        return self.name
