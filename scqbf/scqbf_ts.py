import math
import numpy as np
from scqbf.scqbf_instance import *
from scqbf.scqbf_evaluator import *
import random
import time
from collections import deque
from typing import Literal

PLACE_HOLDER = -1

class ScQbfTS():
    
    def __init__(self, instance: ScQbfInstance, tenure: int, max_iter: int = None, time_limit_secs: int = None, patience: int = None,
                 search_strategy: Literal['first', 'best'] = 'first', debug: bool = False, save_history: bool = False):
        
        # TS-related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)
        self.tabu_list = deque([PLACE_HOLDER] * tenure * 2, maxlen=tenure*2)  # Tabu list implemented as a deque with fixed max length
        self.best_solution: ScQbfSolution = None
        self.current_solution: ScQbfSolution = None
        
        # Config properties
        self.search_strategy = search_strategy 

        # Termination criteria properties
        self._prev_best_solution = None
        self.max_iter = max_iter
        self.time_limit_secs = time_limit_secs
        self.patience = patience
        self._iter = 0
        self._start_time = None
        self._no_improvement_iter = 0
        self.stop_reason: str = None
        
        # Other
        self.debug = debug
        self.save_history = save_history
        if self.save_history:
            self.history = []

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
            elif self.best_solution is not None and self.current_solution is not None and self._prev_best_solution is not None:
                if self.evaluator.evaluate_objfun(self.current_solution) <= self.evaluator.evaluate_objfun(self._prev_best_solution):
                    self._no_improvement_iter += 1
                else:
                    self._no_improvement_iter = 0

        self._prev_best_solution = self.best_solution

        return False
    
    def _do_iteration_internal_actions(self):
        if self.debug:
            print(f"Iteration {self._iter}: Best ObjFun = {self.evaluator.evaluate_objfun(self.best_solution)}, Current ObjFun = {self.evaluator.evaluate_objfun(self.current_solution)}")

        if self.save_history:
            self.history.append((self._iter, self.evaluator.evaluate_objfun(self.best_solution), self.evaluator.evaluate_objfun(self.current_solution)))
    
    def solve(self) -> ScQbfSolution:
        self.best_solution = self._constructive_heuristic()
        best_solution_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)

        self.current_solution = self.best_solution
        current_solution_objfun_val = best_solution_objfun_val
        
        self._start_time = time.time()
        self._iter = 0
        while not self._eval_termination_condition():
            self._do_iteration_internal_actions()
            
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
        if self.search_strategy == 'first':
            return self._neighborhood_move_first_improving(solution)
        elif self.search_strategy == 'best':
            return self._neighborhood_move_best_improving(solution)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
        
    def _neighborhood_move_best_improving(self, solution: ScQbfSolution) -> ScQbfSolution:
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
                    best_delta = delta
                    best_cand_in = cand_in
                    best_cand_out = None
        
        # Evaluate removals
        for cand_out in solution.elements:
            delta = self.evaluator.evaluate_removal_delta(cand_out, solution)  
            
            aspiration_criterion = current_objfun_val + delta > best_objfun_val
            if cand_out not in self.tabu_list or aspiration_criterion:
                if delta > best_delta:
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(solution.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_feasible(temp_sol):
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
                        temp_sol.elements.append(cand_in)
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_feasible(temp_sol):
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

    def _neighborhood_move_first_improving(self, solution: ScQbfSolution) -> ScQbfSolution:
        selected_cand_in = None
        selected_cand_out = None
        
        cl = [i for i in range(self.instance.n) if i not in solution.elements] 
        current_objfun_val = self.evaluator.evaluate_objfun(solution)
        best_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)
        
        improvement_found = False
        
        def evaluate_insertions():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_in in cl:
                delta = self.evaluator.evaluate_insertion_delta(cand_in, solution) 
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if cand_in not in self.tabu_list or aspiration_criterion:
                    if delta > 0:
                        selected_cand_in = cand_in
                        selected_cand_out = None
                        improvement_found = True
                        return
        
        def evaluate_removals():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_out in solution.elements:
                delta = self.evaluator.evaluate_removal_delta(cand_out, solution)  
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if cand_out not in self.tabu_list or aspiration_criterion:
                    if delta > 0:
                        # Check if removing this element would break feasibility
                        temp_sol = ScQbfSolution(solution.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_feasible(temp_sol):
                            selected_cand_in = None
                            selected_cand_out = cand_out
                            improvement_found = True
                            return
        
        def evaluate_exchanges():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_in in cl:
                for cand_out in solution.elements:
                    delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, solution)  
                    
                    aspiration_criterion = current_objfun_val + delta > best_objfun_val
                    if (cand_in not in self.tabu_list and cand_out not in self.tabu_list) or aspiration_criterion:
                        if delta > 0:
                            # Check if removing this element would break feasibility
                            temp_sol = ScQbfSolution(solution.elements.copy())
                            temp_sol.elements.append(cand_in)
                            temp_sol.elements.remove(cand_out)
                            if self.evaluator.is_solution_feasible(temp_sol):
                                selected_cand_in = cand_in
                                selected_cand_out = cand_out
                                improvement_found = True
                                return
        
        actions = [evaluate_insertions, evaluate_removals, evaluate_exchanges]
        random.shuffle(actions)
        for next_action in actions:
            if not improvement_found:
                next_action()
            else:
                if self.debug:
                    print(f"First improving move found with action {next_action.__name__}")
                break
            
        
        new_solution = ScQbfSolution(solution.elements.copy())
        
        # Implement the best move and update tabu list
        if selected_cand_out is not None:
            new_solution.elements.remove(selected_cand_out)
            self.tabu_list.append(selected_cand_out)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        if selected_cand_in is not None:
            new_solution.elements.append(selected_cand_in)
            self.tabu_list.append(selected_cand_in)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        return new_solution
    
        