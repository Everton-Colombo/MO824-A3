import math
import numpy as np
from scqbf.scqbf_instance import *
from scqbf.scqbf_evaluator import *
import random
import time
from collections import deque

PLACE_HOLDER = -1

class ScQbfTS():
    
    def __init__(self, instance: ScQbfInstance, tenure: int, max_iter: int = None, time_limit_secs: int = None, patience: int = None):
        
        # Problem-related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)
        self.tabu_list = deque([PLACE_HOLDER] * tenure * 2, maxlen=tenure*2)  # Tabu list implemented as a deque with fixed max length
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
        best_delta = float('-inf')
        best_cand_in = None
        best_cand_out = None
        
        cl = [i for i in range(self.instance.n) if i not in solution.elements] 
        current_objfun_val = self.evaluator.evaluate_objfun(solution)
        best_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)
        
        # Evaluate insertions
        for cand_in in cl:
            delta = self.evaluator.evaluate_insertion_delta(cand_in, solution) 
            
            aspiration_criterion = current_objfun_val + delta > best_objfun_val
            if cand_in not in self.tabu_list or aspiration_criterion:
                if delta > best_delta:
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(solution.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_valid(temp_sol):
                        best_delta = delta
                        best_cand_in = None
                        best_cand_out = cand_out
        
        # Evaluate removals
        for cand_out in solution.elements:
            delta = self.evaluator.evaluate_removal_delta(cand_out, solution)  
            
            aspiration_criterion = current_objfun_val + delta > best_objfun_val
            if cand_out not in self.tabu_list or aspiration_criterion:
                if delta > best_delta:
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(solution.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_valid(temp_sol):
                        best_delta = delta
                        best_cand_in = None
                        best_cand_out = cand_out
        
        # Evaluate exchanges
        for cand_in in cl:
            for cand_out in solution.elements:
                delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, solution)  
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if (cand_in not in self.tabu_list and cand_out not in self.tabu_list) or aspiration_criterion:
                    if delta > best_delta:
                        # Check if removing this element would break feasibility
                        temp_sol = ScQbfSolution(solution.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_valid(temp_sol):    
                            best_delta = delta
                            best_cand_in = cand_in
                            best_cand_out = cand_out
        
        new_solution = ScQbfSolution(solution.elements.copy())
        
        # Implement the best move and update tabu list
        if best_cand_out is not None:
            new_solution.elements.remove(best_cand_out)
            self.tabu_list.append(best_cand_out)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        if best_cand_in is not None:
            new_solution.elements.append(best_cand_in)
            self.tabu_list.append(best_cand_in)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        return new_solution
    
        