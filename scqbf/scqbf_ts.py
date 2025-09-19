import math
import numpy as np
from scqbf.scqbf_instance import *
from scqbf.scqbf_evaluator import *
import random
import time

class ScQbfTS():
    
    def __init__(self, instance: ScQbfInstance, tenure: int, max_iter: int = None, time_limit_secs: int = None, patience: int = None):
        
        # Problem-related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)
        self.tenure = tenure
        self.best_solution: ScQbfSolution = None
        self.current_solution: ScQbfSolution = None

        # Termination criteria properties
        self.max_iter = max_iter
        self.time_limit_secs = time_limit_secs
        self.patience = patience
        self._iter = 0
        self._start_time = None
        self._no_improvement_iter = 0
        self.stop_reason: str = None
        

    def _eval_termination_condition(self) -> bool:
        """ Check if the termination condition is met, while also managing termination criteria properties."""

        self._iter += 1
        if self.max_iter is not None and self._iter >= self.max_iter:
            self.stop_reason = "max_iter"
            return True
        if self.time_limit_secs is not None and (time.time() - self._start_time) >= self.time_limit_secs:
            self.stop_reason = "time_limit"
            return True
        if self.patience is not None:
            if self._no_improvement_iter >= self.patience:
                self.stop_reason = "patience_exceeded"
                return True
            elif self.best_solution is not None and self.current_solution is not None:
                if self.evaluator.evaluate_objfun(self.best_solution) > self.evaluator.evaluate_objfun(self.current_solution):
                    self._no_improvement_iter += 1
                else:
                    self._no_improvement_iter = 0
        
        return False
    
    def solve(self) -> ScQbfSolution:
        self.best_solution = self._constructive_heuristic()
        best_solution_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)

        self.current_solution = self.best_solution
        current_solution_objfun_val = best_solution_objfun_val
        
        self._start_time = time.time()
        self._iter = 0
        while not self._eval_termination_condition():
            self.current_solution = self._neighborhood_move(self.current_solution)
            current_solution_objfun_val = self.evaluator.evaluate_objfun(self.current_solution)

            if current_solution_objfun_val > best_solution_objfun_val:
                self.best_solution = self.current_solution
                best_solution_objfun_val = current_solution_objfun_val

        return self.best_solution
    
    def _constructive_heuristic(self) -> ScQbfSolution:
        """
        Very simple constructive heuristic. Adds elements that add coverage until solution is feasible.
        """
        
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)]
        random.shuffle(cl)
        
        while not self.evaluator.is_solution_feasible(constructed_sol):
            rcl = [i for i in cl if i not in constructed_sol.elements 
                      and self.evaluator.evaluate_insertion_delta_coverage(i, constructed_sol) > 0]

            element = random.choice(rcl)
            rcl.remove(element)
            constructed_sol.elements.append(element)
            
            cl = rcl

        if not self.evaluator.is_solution_feasible(constructed_sol):
            raise ValueError("Constructive heuristic failed to produce a feasible solution")
        
        return constructed_sol

    def _neighborhood_move(self, solution: ScQbfSolution) -> ScQbfSolution:
        pass
    
        