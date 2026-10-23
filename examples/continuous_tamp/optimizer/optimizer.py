import numpy as np
import random

from examples.continuous_tamp.primitives import BLOCK_WIDTH, plan_motion, GRASP
from pddlstream.language.constants import partition_facts, NOT, MINIMIZE, is_parameter
from pddlstream.language.optimizer import OptimizerOutput
from pddlstream.utils import INF

MIN_CLEARANCE = 1e-3 # 0 | 1e-3

def has_gurobi():
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True

def value_from_var(vars):
    import gurobipy
    if isinstance(vars, float):
        return vars
    if isinstance(vars, gurobipy.Var):
        return vars.X # .Xn
    new_vars = list(map(value_from_var, vars))
    if isinstance(vars, np.ndarray):
        return np.array(new_vars)
    return new_vars

def unbounded_var(model):
    from gurobipy import GRB
    return model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)

def np_var(model, d=2, lower=None, upper=None):
    from gurobipy import GRB
    if lower is None:
        lower = d*[-GRB.INFINITY]
    if upper is None:
        upper = d*[+GRB.INFINITY]
    return np.array([model.addVar(lb=lb, ub=ub) for lb, ub in zip(lower, upper)])

def copy_model(model):
    model.update()
    return model.copy()

def vars_from_expr(expr):
    return [expr.getVar(i) for i in range(expr.size())]

def set_guess(var, value, hard=True):
    if hard:
        # solver will try to build a single feasible solution from the provided set of variable values
        var.Start = value
    else:
        # variable hints provide guidance to the MIP solver that affects the entire solution process
        var.VarHintVal = value

##################################################

def collision_constraint(model, name, b1, p1, b2, p2):
    from gurobipy import GRB
    dist = unbounded_var(model)
    abs_dist = unbounded_var(model)
    model.addLConstr(dist, GRB.EQUAL, p2[0] - p1[0], name=name)
    model.addLConstr(BLOCK_WIDTH + MIN_CLEARANCE, GRB.LESS_EQUAL, abs_dist, name=name)
    model.addGenConstrAbs(abs_dist, dist, name=name)  # abs_

def kinematics_constraint(model, name, b, q, p, g):
    from gurobipy import GRB
    for i in range(len(q)):
        model.addLConstr(q[i] + g[i], GRB.EQUAL, p[i], name=name)

def contained_constraint(model, regions, name, b, p, r):
    from gurobipy import GRB
    px, py = p
    x1, x2 = regions[r]
    model.addLConstr(x1, GRB.LESS_EQUAL, px - BLOCK_WIDTH / 2, name=name)
    model.addLConstr(px + BLOCK_WIDTH / 2, GRB.LESS_EQUAL, x2, name=name)
    model.addLConstr(py, GRB.EQUAL, 0, name=name)

def distance_cost(model, q1, q2, norm=1):
    from gurobipy import GRB
    # TODO: cost on endpoints and subtract from total cost
    terms = []
    for i in range(len(q1)):
        delta = q2[i] - q1[i]
        if norm == 1:
            # TODO: doesn't work with deletion filter (additional constraints)
            distance = model.addVar(lb=0., ub=GRB.INFINITY)
            model.addConstr(-delta <= distance)
            model.addConstr(delta <= distance)
            terms.append(distance)
        elif norm == 2:
            terms.append(delta * delta)
        else:
            raise RuntimeError(norm)
    return terms

##################################################

##################################################

def sample_targets(model, variables):
    # TODO: project without inequality constraints (placement)
    # TODO: manifold learning for feasible subspace
    # TODO: sample from neighborhood of the previous solution
    from gurobipy import GRB
    model.update()
    objective_terms = []
    for var in variables:
        # TODO: apply only to some variables
        if (-INF < var.LB) and (var.UB < INF):
            # TODO: inequality constraint version of this
            extent = var.UB - var.LB
            value = random.uniform(var.LB, var.UB)
            delta = var - value
            distance = model.addVar(lb=0., ub=GRB.INFINITY)
            model.addConstr(-delta <= distance)
            model.addConstr(delta <= distance)
            objective_terms.append(distance / extent)
            print(var, var.LB, var.UB, value)
            # TODO: start with feasible solution and then expand
    return objective_terms

##################################################

def get_optimize_fn(regions, collisions=True, max_time=5., hard=False,
                    diagnostic='all', diagnose_cost=False, verbose=False):
    # https://www.gurobi.com/documentation/8.1/examples/diagnose_and_cope_with_inf.html
    # https://www.gurobi.com/documentation/8.1/examples/tsp_py.html#subsubsection:tsp.py
    # https://examples.xpress.fico.com/example.pl?id=Infeasible_python
    # https://www.fico.com/fico-xpress-optimization/docs/latest/examples/python/GUID-E77AC35C-9488-3F0A-98B4-7F5FD81AFF1D.html
    if not has_gurobi():
        raise ImportError('This generator requires Gurobi: http://www.gurobi.com/')
    from gurobipy import Model, GRB, quicksum

    min_x = min(x1 for x1, _ in regions.values())
    max_x = max(x2 for _, x2 in regions.values())
    min_y, max_y = 0, (max_x - min_x) / 2.
    lower = [min_x, min_y]
    upper = [max_x, max_y]


    def fn(outputs, facts, hint={}):
        # TODO: pass in the variables and constraint streams instead?
        # The true test is placing two blocks in a tight region obstructed by one
        constraint_indices = {i for i, term in enumerate(facts) if term[0] != MINIMIZE}
        positive, negative, costs = partition_facts(facts)
        if costs:
            print('Costs:', costs)

        # https://github.com/yijiangh/coop_assembly/blob/e52abef7c1cfb1d3e32691d163abc85dd77f27a2/src/coop_assembly/geometry_generation/caelan.py
        model = Model(name='TAMP')
        model.setParam(GRB.Param.OutputFlag, verbose)
        model.setParam(GRB.Param.TimeLimit, max_time)
        model.setParam(GRB.Param.Cutoff, GRB.INFINITY) # TODO: account for scaling


        ##########

        # TODO: remove anything that's just a domain condition?
        variable_indices = {}
        var_from_param = {}
        for index, fact in enumerate(facts):
            prefix, args = fact[0], fact[1:]
            if prefix == 'conf':
                param, = args
                if is_parameter(param):
                    var_from_param[param] = np_var(model, lower=lower, upper=upper)
            elif prefix == 'pose':
                _, param = args
                if is_parameter(param):
                    var_from_param[param] = np_var(model, lower=lower, upper=upper)
            elif prefix == 'grasp':  # TODO: iterate over combinations
                _, param = args
                if is_parameter(param):
                    var_from_param[param] = GRASP
            elif prefix == 'traj':
                raise NotImplementedError()
            else:
                continue
            variable_indices[index] = fact
        dimension = sum(len(var) for var in var_from_param.values())

        def get_var(p):
            return var_from_param[p] if is_parameter(p) else p

        ##########

        codimension = 0
        objective_terms = [] # TODO: could make a variable to impose a cost constraint
        constraint_from_name = {}
        for index, fact in enumerate(facts):
            prefix, args = fact[0], fact[1:]
            name = str(index)
            if prefix == 'kin':
                kinematics_constraint(model, name, *map(get_var, args))
                codimension += 2
            elif prefix in ('contain', 'contained'):
                contained_constraint(model, regions, name, *map(get_var, args))
                codimension += 1
            elif prefix == 'cfree' and collisions:
                # TODO: drop collision constraints until violated
                collision_constraint(model, name, *map(get_var, args))
            elif prefix == 'motion':
                raise NotImplementedError()
            elif prefix == NOT:
                fact = args[0]
                predicate, args = fact[0], fact[1:]
                if predicate == 'posecollision' and collisions:
                    collision_constraint(model, name, *map(get_var, args))
            elif prefix == MINIMIZE:
                fact = args[0]
                func, args = fact[0], fact[1:]
                if func in ('dist', 'distance'):
                    objective_terms.extend(distance_cost(model, *map(get_var, args)))
                continue
            constraint_from_name[name] = fact
        model.update()

        ##########

        linear_model = copy_model(model)

        # TODO: prune linearly dependent constraints
        linear_constraints = {c for c in linear_model.getConstrs() if c.sense == GRB.EQUAL}
        codimension = len(linear_constraints)
        linear_variables = set()
        for c in linear_constraints:
            linear_variables.update(vars_from_expr(linear_model.getRow(c)))
        linear_variables = sorted(linear_variables, key=lambda v: v.VarName)
        dimension = len(linear_variables)

        print('{} variables (dim={}): {}'.format(len(variable_indices), dimension,
                                                 [facts[index] for index in sorted(variable_indices)]))
        nontrivial_indices = set(constraint_indices) - set(variable_indices) # TODO: rename
        print('{} constraints: (codim={}): {}'.format(len(nontrivial_indices), codimension,
                                                      [facts[index] for index in sorted(nontrivial_indices)]))



        ##########

        # TODO: normalize cost relative to the best cost for a trade-off
        # TODO: increasing bound on deterioration in quality
        weight = 0
        if weight > 0:
            primary_variables = {v for var in var_from_param.values() for v in var}
            objective_terms.extend(weight * term for term in sample_targets(model, primary_variables))
        model.setObjective(quicksum(objective_terms), sense=GRB.MINIMIZE) # (1-weight) * quicksum(objective_terms)

        for out, value in hint.items():
            for var, coord in zip(get_var(out), value):
                # https://www.gurobi.com/documentation/9.1/refman/varhintval.html#attr:VarHintVal
                set_guess(var, coord, hard=hard)

        ##########

        model.optimize()
        if model.SolCount == 0:
            if diagnostic is None:
                return OptimizerOutput()
            elif diagnostic == 'all':
                infeasible = nontrivial_indices
            elif diagnostic == 'deletion':
                infeasible = deletion_filter(model, constraint_indices)
            elif diagnostic == 'elastic':
                infeasible = elastic_filter(model, constraint_indices)
            elif diagnostic == 'gurobi':
                infeasible = compute_inconsistent(model)
            else:
                raise NotImplementedError(diagnostic)
            print('Inconsistent:', [facts[index] for index in sorted(infeasible)])
            return OptimizerOutput(infeasible=[infeasible])


        ##########

        print('Solved: {} | Objective: {:.3f} | Solutions: {} | Status: {} | Runtime: {:.3f}'.format(
            True, model.ObjVal, model.SolCount, model.status, model.runtime))

        if costs and diagnose_cost:
            infeasible = deletion_filter(model, constraint_indices, max_objective=model.ObjVal - 1e-6)
        else:
            # TODO: propagate automatically to optimizer
            infeasible = nontrivial_indices
        print('Cost inconsistent:', [facts[index] for index in sorted(infeasible)])


        assignment = tuple(value_from_var(get_var(out)) for out in outputs)
        return OptimizerOutput(assignments=[assignment], infeasible=[infeasible])
    return fn

##################################################

def compute_inconsistent(model):
    from gurobipy import GRB
    # TODO: search over irreducible infeasible sets
    model.setObjective(0.0) # Makes a difference in performance
    model.setParam(GRB.Param.IISMethod, 1) # -1 | 0 | 1 | 2 | 3

    model.computeIIS()
    print('IIS is minimal\n' if model.IISMinimal else 'IIS is not minimal\n')
    infeasible = {int(c.constrName) for c in model.getConstrs() if c.IISConstr}
    return infeasible

##################################################

def constraints_from_indices(model, indices):
    names = {str(i) for i in indices}
    return [c for c in model.getConstrs() if c.constrName in names]

def deletion_filter(original_model, indices, max_objective=INF):
    from gurobipy import GRB

    model = copy_model(original_model)
    if max_objective < INF:
        model.setParam(GRB.Param.Cutoff, max_objective)
    else:
        model = model.feasibility()

    model.optimize()
    assert model.SolCount == 0

    inconsistent = set()
    for index in sorted(indices):
        temp_model = copy_model(model)
        constraints = constraints_from_indices(temp_model, {index})
        if not constraints:
            continue
        for c in constraints:
            temp_model.remove(c)
        temp_model.optimize()
        if temp_model.SolCount == 0:
            model = temp_model
        else:
            inconsistent.add(index)
    return inconsistent

##################################################

def relax_constraints(model, indices):
    from gurobipy import GRB
    # http://www.sce.carleton.ca/faculty/chinneck/docs/ChinneckDravnieks.pdf
    model.setObjective(0.0)
    elastic_constraints = constraints_from_indices(model, indices)
    model.feasRelax(relaxobjtype=0, # feasRelaxS
                                minrelax=True,
                                vars=[], lbpen=[], ubpen=[],
                                constrs=elastic_constraints,
                                rhspen=[1.]*len(elastic_constraints))
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        return None
    art_vars = [var for var in model.getVars()
                if any(var.varname.startswith(p) for p in ['ArtP_', 'ArtN_'])] # Positive & Negative
    violations = {}
    for var in art_vars:
        index = int(var.varname[5:])
        violations[index] = violations.get(index, 0) + var.x
    violated = {index for index, value in violations.items() if 1e-6 < value}
    return violated

def elastic_filter(original_model, indices, postproces=True):
    elastic = set(indices)
    inconsistent = set()
    while True:
        relaxed_model = copy_model(original_model)
        violated = relax_constraints(relaxed_model, elastic)
        if violated is None:
            break
        elastic -= violated
        inconsistent |= violated
    if postproces:
        inconsistent = deletion_filter(original_model, inconsistent)
    return inconsistent

##################################################

##################################################

def cfree_motion_fn(outputs, facts, hint={}):
    if not outputs:
        return None
    assert(len(outputs) == 1)
    # TODO: handle connected components
    q0, q1 = None, None
    placed = {}
    for fact in facts:
        if fact[0] == 'motion':
            if q0 is not None:
                return None
            q0, _, q1 = fact[1:]
        if fact[0] == NOT:
            _, b, p =  fact[1][1:]
            placed[b] = p
    if q0 is None:
        t = []
        return (t,)
    return plan_motion(q0, q1)

##################################################

