from typing import List, Tuple, Dict

import numpy as np
import gurobipy.gurobipy as grb


num_machines = 2
num_tasks = 7

weights = np.array([
    [4, 1, 2, 1, 4, 3, 8],
    [9, 9, 8, 1, 3, 8, 7]
])

profits = np.array([
    [6, 9, 4, 2, 10, 3, 6],
    [4, 8, 9, 1, 7, 5, 4]
])

capacity = np.array([11, 22])

initial_solution = [
    (0, [0, 1, 2, 5]),
    (1, [3, 4, 6])
]

model_type = grb.GRB.MAXIMIZE
#model_type = grb.GRB.MINIMIZE
print(f"Solving MINIMIZE")

def machine_schedule_profit(machine_schedule) -> float:
    machine_id = machine_schedule[0]
    tasks = machine_schedule[1]
    return profits[machine_id][tasks].sum()


def build_task_constraints(model) -> Dict:
    task_to_constraint = dict()
    for task_id in range(num_tasks):
        lhs = grb.quicksum([])
        rhs = 1
        name = f'task_assignment_{task_id}'
        c = model.addConstr(lhs == rhs, name=name)
        task_to_constraint[task_id] = c
    return task_to_constraint


def build_convexity_constraints(model) -> Dict:
    machine_to_constraint = dict()
    for machine_id in range(num_machines):
        lhs = grb.quicksum([])
        rhs = 1
        name = f'convexity_machine_{machine_id}'
        c = model.addConstr(lhs == rhs, name=name)
        machine_to_constraint[machine_id] = c
    return machine_to_constraint


def get_column(machine_schedule, machine_to_constraint, task_to_constraint) -> grb.Column:
    machine_id = machine_schedule[0]
    tasks = machine_schedule[1]

    c = grb.Column()
    coeff = 1.0
    for task in tasks:
        constr = task_to_constraint[task]
        c.addTerms(coeff, constr)

    machine_constr = machine_to_constraint[machine_id]
    c.addTerms(coeff, machine_constr)
    return c


def model_1():
    model = grb.Model("GAP_variables_with_upper_bound")
    model.setAttr(grb.GRB.Attr.ModelSense, model_type)
    model.Params.LogToConsole = 0
    task_to_constraint = build_task_constraints(model)
    machine_to_constraint = build_convexity_constraints(model)

    for machine_schedule in initial_solution:
        machine_id = machine_schedule[0]
        tasks = machine_schedule[1]
        profit = machine_schedule_profit(machine_schedule)

        name = f"machine_{machine_id}_tasks_{'_'.join(str(task_id) for task_id in tasks)}"
        column = get_column(machine_schedule, machine_to_constraint, task_to_constraint)

        _var = model.addVar(
            lb=0.0,
            ub=1.0,
            obj=profit,
            vtype=grb.GRB.CONTINUOUS,
            name=name,
            column=column
        )

    model.update()
    model.write(model.ModelName + '.lp')
    model.optimize()
    print(f"Model >{model.ModelName}< objective value: {model.ObjVal}")
    print(f"Model >{model.ModelName}< duals: {model.Pi}")
    print(f"Model >{model.ModelName}< duals: {model.RC}")

    for var in model.getVars():
        print(f"Var >{var.VarName}< duals: {var.RC}")


def model_2():
    model = grb.Model("GAP_variables_with_no_upper_bound")
    model.setAttr(grb.GRB.Attr.ModelSense, model_type)
    model.Params.LogToConsole = 0
    task_to_constraint = build_task_constraints(model)
    machine_to_constraint = build_convexity_constraints(model)

    for machine_schedule in initial_solution:
        machine_id = machine_schedule[0]
        tasks = machine_schedule[1]
        profit = machine_schedule_profit(machine_schedule)

        name = f"machine_{machine_id}_tasks_{'_'.join(str(task_id) for task_id in tasks)}"
        column = get_column(machine_schedule, machine_to_constraint, task_to_constraint)

        _var = model.addVar(
            lb=0.0,
            # ub=1.0,
            obj=profit,
            vtype=grb.GRB.CONTINUOUS,
            name=name,
            column=column
        )

    model.update()
    model.write(model.ModelName + '.lp')
    model.optimize()
    print(f"Model >{model.ModelName}< objective value: {model.ObjVal}")
    print(f"Model >{model.ModelName}< duals: {model.Pi}")


def model_3():
    model = grb.Model("GAP_variables_with_upper_bound_as_constraint")
    model.setAttr(grb.GRB.Attr.ModelSense, model_type)
    model.Params.LogToConsole = 0
    task_to_constraint = build_task_constraints(model)
    machine_to_constraint = build_convexity_constraints(model)

    for machine_schedule in initial_solution:
        machine_id = machine_schedule[0]
        tasks = machine_schedule[1]
        profit = machine_schedule_profit(machine_schedule)

        name = f"machine_{machine_id}_tasks_{'_'.join(str(task_id) for task_id in tasks)}"
        column = get_column(machine_schedule, machine_to_constraint, task_to_constraint)

        var = model.addVar(
            lb=0.0,
            # ub=1.0,
            obj=profit,
            vtype=grb.GRB.CONTINUOUS,
            name=name,
            column=column
        )

        _c = model.addConstr(var <= 1.0, name=f'Upper_bound_{name}')

    model.update()
    model.write(model.ModelName + '.lp')
    model.optimize()
    print(f"Model >{model.ModelName}< objective value: {model.ObjVal}")
    print(f"Model >{model.ModelName}< duals: {model.Pi}")

model_1()
model_2()
model_3()
