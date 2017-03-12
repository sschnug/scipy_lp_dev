import os
import time
import dataset
from constants import *
from mps_reader_preprocessor import read_mps_preprocess
from solvers import *
from evaluation import *


class Benchmark(object):
    def __init__(self, instance_names=[], solver_names=[], obj_tol=1e-3,
                       mem_lim=4000, time_lim=600, verbose=2):
        # TODO: limit cores?
        # TODO: no limits applied yet

        self.instance_names = instance_names
        self.solver_names = solver_names
        self.obj_tol = obj_tol
        self.mem_lim = mem_lim
        self.time_lim = time_lim
        self.cwd = os.getcwd()
        self.verbose = verbose

        self.db = dataset.connect('sqlite:///results.db')
        self.db_table = self.db['eval_big']

    def run(self):
        for instance_name in self.instance_names:
            if self.verbose:
                print('---')
                print('instance_name: ', instance_name)
                print('---')
            instance_path = os.path.join(self.cwd, NETLIB_INSTANCES_PATH, instance_name) + '.mps'
            problem = read_mps_preprocess(instance_path)
            expected_obj = float(NETLIB_INSTANCES_TO_BENCHMARK[instance_name])

            for solver_name in self.solver_names:
                self.run_single_solve(instance_name, problem, solver_name, expected_obj)

    def run_single_solve(self, instance_name, problem, solver, exp_obj):
        if solver == 'linprog_simplex':
            if self.verbose:
                print()
                print('linprog_simplex')

            start_time = time.perf_counter()
            status, obj = solve_linprog_simplex(problem)
            end_time = time.perf_counter()

            if self.verbose >= 2:
                print(status, obj)
                print(end_time - start_time)
            time_used = end_time - start_time

            valid_result = False
            if abs(obj - exp_obj) <= self.obj_tol:
                valid_result = True
            self.db_table.insert(dict(instance=instance_name,
                                      solver=solver,
                                      valid_result=valid_result,
                                      time_used=time_used))
            #self.results.append((instance_name, solver, valid_result, time_used))

        elif solver == 'linprog_ip':
            if self.verbose:
                print()
                print('linprog_ip')

            start_time = time.perf_counter()
            status, obj = solve_linprog_ip(problem)
            end_time = time.perf_counter()
            time_used = end_time - start_time

            if self.verbose >= 2:
                print(status, obj)
                print(end_time - start_time)

            valid_result = False
            if abs(obj - exp_obj) <= self.obj_tol:
                valid_result = True
            self.db_table.insert(dict(instance=instance_name,
                                      solver=solver,
                                      valid_result=valid_result,
                                      time_used=time_used))
            #self.results.append((instance_name, solver, valid_result, time_used))


        elif solver == 'cvxopt_conelp':
            if self.verbose:
                print()
                print('cvxopt_conelp')

            start_time = time.perf_counter()
            status, obj = solve_cvxopt_conelp(problem)
            end_time = time.perf_counter()
            time_used = end_time - start_time

            if self.verbose >= 2:
                print(status, obj)
                print(end_time - start_time)

            valid_result = False
            if abs(obj - exp_obj) <= self.obj_tol:
                valid_result = True
            self.db_table.insert(dict(instance=instance_name,
                                      solver=solver,
                                      valid_result=valid_result,
                                      time_used=time_used))
            #self.results.append((instance_name, solver, valid_result, time_used))

        elif solver == 'cvxopt_mosek':
            if self.verbose:
                print()
                print('cvxopt_mosek')

            start_time = time.perf_counter()
            status, obj = solve_cvxopt_mosek(problem)
            end_time = time.perf_counter()
            time_used = end_time - start_time

            if self.verbose >= 2:
                print(status, obj)
                print(end_time - start_time)

            valid_result = False
            if abs(obj - exp_obj) <= self.obj_tol:
                valid_result = True
            self.db_table.insert(dict(instance=instance_name,
                                      solver=solver,
                                      valid_result=valid_result,
                                      time_used=time_used))
            #self.results.append((instance_name, solver, valid_result, time_used))

        elif solver == 'simpleIPM':
            if self.verbose:
                print()
                print('simpleIPM')

            start_time = time.perf_counter()
            status, obj = solve_simpleIPM(problem)
            end_time = time.perf_counter()
            time_used = end_time - start_time

            if self.verbose >= 2:
                print(status, obj)
                print(end_time - start_time)

            valid_result = False
            if abs(obj - exp_obj) <= self.obj_tol:
                valid_result = True
            self.db_table.insert(dict(instance=instance_name,
                                      solver=solver,
                                      valid_result=valid_result,
                                      time_used=time_used))
            #self.results.append((instance_name, solver, valid_result, time_used))

""" Some benchmark-utils """
all_ = list(NETLIB_INSTANCES_TO_BENCHMARK.keys())
for i in all_:
    try:
        f = open(os.path.join(NETLIB_INSTANCES_PATH, i) + '.mps')
    except:
        print('Problem with: ', i)
        assert False

""" Run Benchmark """
benchmark = Benchmark(all_, \
                      ['linprog_simplex', 'linprog_ip'])                     # no mosek for now
benchmark.run()

""" Evaluation """
# df = data_wrangling(benchmark.results)
# plot_overview(df)
