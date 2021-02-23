import pyomo.environ as pe
import import_ipynb
from pyomo.core.base.block import generate_cuid_names
import random
import time
from optmodel_utilities import *
#from BoxConstrainedGO_Algorithms import MBH, multistart, purerandomsearch
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


# Define local path and function for loading solvers
path = "C:/Users/Pi-He/OneDrive/Bureau/Cours/Optimization/PyomoSolvers_win/solvers/"

def create_solver(solver_name = 'cplex'):
    solver_path = get_solver_path(solver_name)
    return  SolverFactory(solver_name, executable=str(solver_path), solver_io = 'nl')


# Define algorithms for Circle packing problem

# random generating point keeping in [lb,ub]
def random_point(model, gen_multi):
    for i in model.N:
        model.x[i] = gen_multi.uniform(model.lb, model.ub)
        model.y[i] = gen_multi.uniform(model.lb, model.ub)
        model.r = gen_multi.uniform(0.0, 1.0)

# perturbation
def perturb_point(model, gen_pert, delta):
    for i in model.N:
        model.x[i] = model.x[i].value*(1+gen_pert.uniform(-1, 1) * delta)
        model.x[i] = max(model.lb, min(model.x[i].value, model.ub))
        model.y[i] = model.y[i].value * (1 + gen_pert.uniform(-1, 1) * delta)
        model.y[i] = max(model.lb, min(model.y[i].value, model.ub))
        model.r = 0

def new_perturb_point(model, gen_pert):
    xi=0.5/((model.n)**0.5)
    for i in model.N:
        x_lb=max(0.0,model.x[i].value-xi)
        x_ub=min(1.0,model.x[i].value+xi)
        y_lb=max(0.0,model.y[i].value-xi)
        y_ub=min(1.0,model.y[i].value+xi)
        model.x[i] = gen_pert.uniform(x_lb, x_ub)
        model.y[i] = gen_pert.uniform(y_lb, y_ub)
    model.r = 0.0


def check_if_optimal(results):
    ok = (results.solver.status == pe.SolverStatus.ok)
    optimal = (results.solver.termination_condition
               == pe.TerminationCondition.optimal)
    #print(ok)
    #print(optimal)
    #quit()
    return (ok and optimal)

# this function performs multistart on a model and
# for a given number of iterations
def multistart(mymodel, iter, gen_multi, localsolver, labels,
               epsilon, logfile=None):
    algo_name = "Multi:"
    best_obj = 0  # put a reasonable value bound for the objective
    bestpoint = {}  # dictionary to store the best solution

    nb_solution = 0
    feasible = False
    for it in range(1, iter + 1):
        random_point(mymodel, gen_multi)

        # local search
        results = localsolver.solve(mymodel)

        # printing result and solution on screen
        if check_if_optimal(results):
            nb_solution += 1  # couting feasible iterations
            obj = mymodel.obj()
            print(algo_name + " Iteration ", it, " current value ", obj, end='', file=logfile)
            if obj > best_obj + epsilon:  # + epsilon:
                best_obj = obj
                print(" *", file=logfile)
                printPointFromModel(mymodel, logfile)
                feasible = True
                StorePoint(mymodel, bestpoint, labels)
            else:
                print(file=logfile)
        else:
            print(algo_name + " Iteration ", it, "No feasible solution", file=logfile)

    if feasible == True:
        print(algo_name + " Best record found  {0:8.11f}".format(best_obj))
        LoadPoint(mymodel, bestpoint)
        printPointFromModel(mymodel)
    else:
        print(algo_name + " No feasible solution found by local solver")

    print(algo_name + " Total number of feasible solutions ", nb_solution)

    return feasible, bestpoint


def MBH(mymodel, gen, localsolver, labels,
        max_no_improve, pert, delta, epsilon,
        logfile=None):
    algo_name = "MBH:"
    best_obj = 0.5  # put a reasonable value bound for the objective
    bestpoint = {}  # dictionary to store the best solution (and current center)

    feasible = False
    no_improve = 0
    nb_solution = 0

    # look for a starting local solution

    random_point(mymodel, gen)
    results = localsolver.solve(mymodel, load_solutions=True)

    def disp_points():
        print("Global Random Generation")
        print(mymodel.x[1], mymodel.x[1].value)
        print(mymodel.x[2], mymodel.x[2].value)
        print(mymodel.x[3], mymodel.x[3].value)
        print(mymodel.x[4], mymodel.x[4].value)
        print(mymodel.x[5], mymodel.x[5].value)
        print(mymodel.r.value)
        print("\n")
        print("Local optimizer")
        print(mymodel.x[1], mymodel.x[1].value)
        print(mymodel.x[2], mymodel.x[2].value)
        print(mymodel.x[3], mymodel.x[3].value)
        print(mymodel.x[4], mymodel.x[4].value)
        print(mymodel.x[5], mymodel.x[5].value)
        print(mymodel.r.value)

    # a first feasible solution is found
    if check_if_optimal(results):
        nb_solution += 1  # couting feasible iterations
        best_obj = mymodel.obj()
        StorePoint(mymodel, bestpoint, labels)

        print(algo_name, " starting center ", " current value ", best_obj, " *", file=logfile)

        # start local search (perturbation of the current point)
        while (no_improve < max_no_improve):

            perturb_point(mymodel, pert, delta)
            results = localsolver.solve(mymodel, load_solutions=True)
            obj = mymodel.obj()

            def disp_points2():
                print("\n")
                print("Local random Generation")
                print(mymodel.x[1], mymodel.x[1].value)
                print(mymodel.x[2], mymodel.x[2].value)
                print(mymodel.x[3], mymodel.x[3].value)
                print(mymodel.x[4], mymodel.x[4].value)
                print(mymodel.x[5], mymodel.x[5].value)
                print(mymodel.r.value)
                print("\n")

                print("Local optimizer")
                print(mymodel.x[1], mymodel.x[1].value)
                print(mymodel.x[2], mymodel.x[2].value)
                print(mymodel.x[3], mymodel.x[3].value)
                print(mymodel.x[4], mymodel.x[4].value)
                print(mymodel.x[5], mymodel.x[5].value)
                #print(mymodel.r.value)

            if check_if_optimal(results):
                # improving on current center
                #print("current obj: ", obj)
                #print("best obj: ",best_obj)
                if obj > best_obj + epsilon:
                    best_obj = obj

                    print(algo_name + " ", " no_improve ", no_improve, " best_obj ", best_obj, " *", file=logfile)
                    # no load needed - in model there is already the perturbed point
                    no_improve = 0
                    feasible = True
                    StorePoint(mymodel, bestpoint, labels)
                else:
                    # restoring current center
                    LoadPoint(mymodel, bestpoint)
                    print(algo_name + " ", " no_improve ", no_improve, " noImprovingStep", file=logfile)
                    no_improve += 1
                    #print(no_improve)
                #print("new best object: ", best_obj)
    else:
        print(algo_name, " No feasible solution", file=logfile)

    if feasible == True:
        print(algo_name + " Best record found  {0:8.4f}".format(best_obj))
        #mymodel.r.value = best_obj
        LoadPoint(mymodel, bestpoint)
        printPointFromModel(mymodel)
    else:
        print(algo_name + " No feasible solution found by local solver")

    print(algo_name + " Total number of feasible solutions ", nb_solution)

    return feasible, bestpoint


def MBH_MultiTrial(mymodel, iter, gen, localsolver, labels,
                   max_no_improve, pert, delta, epsilon,
                   logfile=None):
    algo_name = "MBH:Trial"
    best_obj = 0.5  # put a reasonable value bound for the objective
    bestpoint = {}  # dictionary to store the best solution
    current_value = best_obj
    currentcenter = {}  # dictionary to store the current center (for pertubation step)

    feasible = False
    for it in range(1, iter + 1):
        no_improve = 0
        nb_solution = 0

        random_point(mymodel, gen)

        # local search
        results = localsolver.solve(mymodel)

        # a first feasible solution is found
        if check_if_optimal(results):
            nb_solution += 1  # couting feasible iterations
            current_value = mymodel.obj()
            StorePoint(mymodel, currentcenter, labels)

            print(algo_name + " ", it, " starting center ", " current value ", current_value, end='', file=logfile)

            # saving GO
            if current_value > best_obj + epsilon:
                best_obj = current_value
                print(" *", file=logfile)
                # printPointFromModel(mymodel)
                feasible = True
                StorePoint(mymodel, bestpoint, labels)
            else:
                print(file=logfile)

            # start local search (perturbation of the current point)
            while (no_improve < max_no_improve):

                perturb_point(mymodel, pert, delta)
                results = localsolver.solve(mymodel)
                obj = mymodel.obj()

                if check_if_optimal(results):
                    # improving on current center
                    if obj > current_value + epsilon:
                        current_value = obj
                        print(algo_name + " Trial ", it, " no_improve ", no_improve, " current value ", current_value,
                              end='', file=logfile)
                        # no load needed - in model there is already the perturbed point
                        no_improve = 0
                        # improving also on global solution
                        if obj > best_obj + epsilon:
                            best_obj = obj
                            print(" *", file=logfile)
                            feasible = True
                            StorePoint(mymodel, bestpoint, labels)
                        else:
                            print(file=logfile)
                    else:
                        # restoring current center
                        LoadPoint(mymodel, currentcenter)
                        print(algo_name + " ", it, " no_improve ", no_improve, " noImprovingStep", file=logfile)
                        no_improve += 1
        else:
            print(algo_name + " ", it, " no_improve ", no_improve, "No feasible solution", file=logfile)

    if feasible == True:
        print(algo_name + " Best record found  {0:8.4f}".format(best_obj))
        LoadPoint(mymodel, bestpoint)
        printPointFromModel(mymodel)
    else:
        print(algo_name + " No feasible solution found by local solver")

    print("Multi:Total number of feasible solutions ", nb_solution)

    return feasible, bestpoint

def CirclePacking(size, lb, ub):

    model = pe.AbstractModel()
    # size
    model.lb = lb
    model.ub = ub
    model.n = pe.Param(default=size)
    # set of variables, useful for sum and iterations
    model.N = pe.RangeSet(model.n)
    model.x = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.y = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.r = pe.Var(bounds=(0.0, 1.0))

    def no_overlap_rule(model, i, j):
        if i < j:
            return(
                (model.x[i] - model.x[j])**2
                + (model.y[i] - model.y[j])**2 >= 4*model.r**2
            )
        else:
            return pe.Constraint.Skip

    model.no_overlap = pe.Constraint(model.N, model.N, rule=no_overlap_rule)

    def Inside_x_min_rule(model, i):
        return model.x[i] >= model.r
    model.Inside_x_min = pe.Constraint(model.N, rule=Inside_x_min_rule)

    def Inside_y_min_rule(model, i):
        return model.y[i] >= model.r
    model.Inside_y_min = pe.Constraint(model.N, rule=Inside_y_min_rule)

    def Inside_x_max_rule(model, i):
        return model.x[i] <= 1-model.r
    model.Inside_x_max = pe.Constraint(model.N, rule=Inside_x_max_rule)

    def Inside_y_max_rule(model, i):
        return model.y[i] <= 1-model.r
    model.Inside_y_max = pe.Constraint(model.N, rule=Inside_y_max_rule)

    def radius_rule(model):
        return model.r

    # then we created the objective: function and sense of optimization
    model.obj = pe.Objective(rule=radius_rule, sense=pe.maximize)
    #model.obj = pe.Objective(rule=radius_rule, sense=pe.minimize)

    model.n = size
    # return instance
    return model.create_instance()


Packomania_radius={'1':0.500000000000000000000000000000,
'2':0.292893218813452475599155637896,
'3':0.254333095030249817754744760429,
'4':0.250000000000000000000000000000,
'5':0.207106781186547524400844362105,
'6':0.187680601147476864319898426192,
'7':0.174457630187009438959427204500,
'8':0.170540688701054438818560595676,
'9':0.166666666666666666666666666667,
'10':0.148204322565228798668007362743,
'11':0.142399237695800384587114500527,
'12':0.139958844038428028961026945453,
'13':0.133993513499008491414263236065,
'14':0.129331793710034021408259201773,
'15':0.127166547515124908877372380214,
'16':0.125000000000000000000000000000,
'17':0.117196742782948687473176894856,
'18':0.115521432463999509608513951182,
'19':0.112265437570996304738752306983,
'20':0.111382347512479750227357863499,
'21':0.106860212352064428580553201716,
'22':0.105665296756976756533092354860,
'23':0.102802323379784112346596984546,
'24':0.101381800431613524388964772877,
'25':0.100000000000000000000000000000,
'26':0.096362339009887092432024697394,
'27':0.095420001747936516364270819029,
'28':0.093672833832785071755016932236,
'29':0.092463144040309496841524670049,
'30':0.091671057985988438718806599233,
'31':0.089338333351234633748802313039,
'32':0.087858157087794923008578158871,
'33':0.087230014135567983530352112436,
'34':0.085270344350527219409280297838,
'35':0.084290712122358459903667877331,
'36':0.083333333333333333333333333333,
'37':0.082089766428752816962342111380,
'38':0.081709776125419800616581793806,
'39':0.081367527046974026631769391643,
'40':0.079186752517282867898915819775,
'41':0.078450210116920559483053007213,
'42':0.077801502898165532306768693062,
'43':0.076339810579744074677200744991,
'44':0.075781986017897705750184232945,
'45':0.074727343686985328535051089524,
'46':0.074272199909420921796026120347,
'47':0.073113151270516836978943966705,
'48':0.072432291280497581081203202077,
'49':0.071692681703623580727975837865,
'50':0.071377103864521467530864106392}


# ### 5.2. Model
# Define the model as circle packing with two points
n = 10
fig = plt.figure(figsize = (18,6))

params=np.linspace(0, 1, 21)
print(params)
time.sleep(5)
#params=[0.1, 0.3, 0.5, 0.7, 0.9]
algo="MBH"
a=0
list_radius=[]
list_pack=[]

for i, v in enumerate(params):
    a+=1
    b = round(len(params) / 3, 0)
    ax = fig.add_subplot(b, 3, a)

    mymodel = CirclePacking(n, 0, 1)

    Multi_n_iter = n * 100
    PRS_n_iter = Multi_n_iter * n * 20
    seed1 = 42

    gen_multi = random.Random()
    print(gen_multi)

    # chosing the solver
    #localsolver = create_solver(path + 'snopt')
    localsolver = create_solver(path + 'knitro')

    print(mymodel)

    # needed to retrieve variable names
    labels = generate_cuid_names(mymodel)

    tech_time = time.process_time()

    logfile = open("mylog.txt", 'w')

    gen_multi.seed(seed1)

    if algo == "Multistart":
        FoundSolution, points = multistart(mymodel, Multi_n_iter, gen_multi, localsolver, labels, logfile)

    elif algo == "MBH":
        FoundSolution, points = MBH(mymodel, gen_multi, localsolver, labels, max_no_improve=n*10, pert=random,
                                    epsilon=10 ** -8, delta=v, logfile=logfile)
    elif algo == "MBH_Multitrial":
        FoundSolution, points = MBH_MultiTrial(mymodel, 100, gen_multi, localsolver, labels, max_no_improve=n * 2,
                                               pert=random, epsilon=10 ** -8, delta=0.1, logfile=logfile)
    print(FoundSolution)
    multistart_time = time.process_time()

    import matplotlib.pyplot as plt
    print(points)
    print(points.keys())

    X=[]
    Y=[]
    for i in points.keys():
        if 'x' in i:
            X.append(points.get(i))
        elif 'y' in i:
            Y.append(points.get(i))


    #x1=points.get('x:#1')
    #x2=points.get('x:#2')
    #y1=points.get('y:#1')
    #y2=points.get('y:#2')
    R=points.get('r')
    list_radius.append(R)
    list_pack.append(Packomania_radius.get(str(n)))

    print(points.values())
    #print(points.get(''))
    import matplotlib.colors

    for i in range(len(X)):
        plt.plot(X[i], Y[i], marker='+')
        circle = plt.Circle((X[i], Y[i]), R, color='gold')
        ax.add_artist(circle)

    plt.title("N=%s (R=%s)"% (n, R))
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
plt.show()


print(list_radius)
print(list_pack)

fig, ax = plt.subplots()
plt.plot(params, list_radius, label=algo)
plt.plot(params, list_pack, label="Packomania")
plt.title("R=f(delta)")
#plt.xlabel("N")
#plt.ylabel("Y")
#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([min(min(list_pack), min(list_radius)), max(max(list_pack), max(list_radius))])
plt.legend()
plt.show()
