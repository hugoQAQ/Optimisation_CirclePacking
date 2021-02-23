import pyomo.environ as pe
import import_ipynb
from pyomo.core.base.block import generate_cuid_names
import random
import time
from optmodel_utilities import *
#from BoxConstrainedGO_Algorithms import MBH, multistart, purerandomsearch
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        model.z[i] = gen_multi.uniform(model.lb, model.ub)
        model.r = gen_multi.uniform(0.0, 1.0)

# perturbation
def perturb_point(model, gen_pert, delta):
    #print('current r: %s' % model.r.value)
    for i in model.N:
        model.x[i] = model.x[i].value*(1+gen_pert.uniform(-1, 1) * delta)
        #project inside the box (ATTENTION: the perturbation is not anymore a uniform distribution)
        model.x[i] = max(model.lb, min(model.x[i].value, model.ub))
        model.y[i] = model.y[i].value * (1 + gen_pert.uniform(-1, 1) * delta)
        model.y[i] = max(model.lb, min(model.y[i].value, model.ub))
        model.z[i] = model.z[i].value * (1 + gen_pert.uniform(-1, 1) * delta)
        model.z[i] = max(model.lb, min(model.z[i].value, model.ub))


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
    best_obj = 0.5  # put a reasonable value bound for the objective
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
        print(algo_name + " Best record found  {0:8.4f}".format(best_obj))
        LoadPoint(mymodel, bestpoint)
        printPointFromModel(mymodel)
    else:
        print(algo_name + " No feasible solution found by local solver")

    print(algo_name + " Total number of feasible solutions ", nb_solution)

    #print(bestpoint)
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

    #Initialisation of the model
    model = pe.AbstractModel()

    # size
    model.lb = lb
    model.ub = ub
    model.n = pe.Param(default=size)

    # set of variables, useful for sum and iterations
    model.N = pe.RangeSet(model.n)
    model.x = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.y = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.z = pe.Var(model.N, bounds=(model.lb, model.ub))
    model.r = pe.Var(bounds=(0.0, 0.5))

    def no_overlap_rule(model, i, j):
        if i < j:
            #print(model.x[i],model.x[i].value)
            #print(model.x[j], model.x[j].value)
            #print(model.r.value)
            return(
                (model.x[i] - model.x[j])**2
                + (model.y[i] - model.y[j])**2
                + (model.z[i] - model.z[j])**2>= 4*model.r**2
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

    def Inside_z_min_rule(model, i):
        return model.z[i] >= model.r
    model.Inside_z_min = pe.Constraint(model.N, rule=Inside_z_min_rule)

    def Inside_z_max_rule(model, i):
        return model.z[i] <= 1-model.r
    model.Inside_z_max = pe.Constraint(model.N, rule=Inside_z_max_rule)


    def radius_rule(model):
        return model.r

    # then we created the objective: function and sense of optimization
    model.obj = pe.Objective(rule=radius_rule, sense=pe.maximize)

    model.n = size
    # return instance
    return model.create_instance()


Packomania_radius={'1':0.500000000000000000000000000000,
'2':0.316987298107780676618138414624,
'3':0.292893218813452475599155637895,
'4':0.292893218813452475599155637895,
'5':0.263932022500210303590826331269,
'6':0.257359312880714853594933827371,
'7':0.250136153876702043245287439440,
'8':0.250000000000000000000000000000,
'9':0.232050807568877293527446341506,
'10':0.214285714285714285714285714286,
'11':0.207622238388108088730418254084,
'12':0.207106785524038967661929205346,
'13':0.207106781186547524400844362105,
'14':0.207106781186547524400844362105,
'15':0.192307692307692307692307692308,
'16':0.188796768447000008806390589866,
'17':0.188685238868306387497346180781,
'18':0.187680601147476864319898426192,
'19':0.183185303161713528511335899512,
'20':0.178407199722670498295402364957,
'21':0.177219044407182163032127767345,
'22':0.173273103227763925774353580229,
'23':0.171816783367860445192850051509,
'24':0.170540688701054438818560595676,
'25':0.167804075634883619081459335660,
'26':0.166905167286093734462460173062,
'27':0.166666666666666666666666666667,
'28':0.160191005918776726965553308878,
'29':0.160188620508520367600361869474,
'30':0.160188620508520367600361869474,
'31':0.160188620508520367600361869474,
'32':0.160188620508520367600361869474,
'33':0.154605431908113458803268257572,
'34':0.152117093470357826262144416884,
'35':0.151668522645211722883250253537,
'36':0.149378180763194405658757891822,
'37':0.149061631334124593491369826337,
'38':0.149038769067752685693797458665,
'39':0.147617379098483843586701989129,
'40':0.147058823529411764705882352941,
'41':0.144837878409323524702584699959,
'42':0.143045425138611931482588665745,
'43':0.141652316450780131858955961430,
'44':0.140860610937643660647157949944,
'45':0.140631937345523308925037746607,
'46':0.140246869718722237549525183737,
'47':0.139958859837643585218718460156,
'48':0.139958844038428028961026945453,
'49':0.136426759340618099952131723523,
'50':0.135954512641663590961947158274}

# ### 5.2. Model
# Define the model as circle packing with two points
algo = "MBH"
l_n = 1
u_n = 50

fig = plt.figure(figsize = (18,6))

a=0
b=1
list_radius=[]
list_pack=[]
for i in range(l_n, u_n+1):
    a+=1
    print("%s Circles"%i)

    b=round((u_n-l_n+1)/3,0)

    ax = fig.add_subplot(b, 3, a, projection='3d')

    mymodel = CirclePacking(i, 0, 1)

    Multi_n_iter = i * 100
    PRS_n_iter = Multi_n_iter * i * 20


    gen_multi = random.Random()
    seed1 = 44

    # chosing the solver
    #localsolver = create_solver(path + 'snopt')
    localsolver = create_solver(path + 'knitro')

    # needed to retrieve variable names
    labels = generate_cuid_names(mymodel)

    tech_time = time.process_time()

    logfile = open("mylog.txt", 'w')

    #gen_multi.seed(seed1)

    if algo == "Multistart":
        FoundSolution, points = multistart(mymodel, Multi_n_iter, gen_multi, localsolver, labels, logfile)

    elif algo == "MBH":
        FoundSolution, points = MBH(mymodel, gen_multi, localsolver, labels, max_no_improve=i * 2, pert=random,
                                    epsilon=10 ** -8, delta=0.3, logfile=logfile)
    elif algo == "MBH_Multitrial":
        FoundSolution, points = MBH_MultiTrial(mymodel, 100, gen_multi, localsolver, labels, max_no_improve=i * 2,
                                               pert=random, epsilon=10 ** -8, delta=0.1, logfile=logfile)
    print(FoundSolution)
    multistart_time = time.process_time()

    import matplotlib.pyplot as plt
    print(points)
    print(points.keys())

    X=[]
    Y=[]
    Z=[]
    for j in points.keys():
        if 'x' in j:
            X.append(points.get(j))
        elif 'y' in j:
            Y.append(points.get(j))
        elif 'z' in j:
            Z.append(points.get(j))

    R=points.get('r')

    print(R)
    list_radius.append(R)
    list_pack.append(Packomania_radius.get(str(i)))
    #print(points.values())

    for j in range(len(X)):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = (np.cos(u) * np.sin(v)) * R + X[j]
        y = (np.sin(u) * np.sin(v)) * R + Y[j]
        z = (np.cos(v) * R + Z[j])
        ax.plot_surface(x, y, z, color="gold")

    plt.title("N=%s (R=%s)"% (i, R))
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.zlabel("Z")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
absc=list(range(l_n, u_n+1))
plt.plot(absc, list_radius, label=algo)
plt.plot(absc, list_pack, label="Packomania")
plt.title("R=f(N)")
#plt.xlabel("N")
#plt.ylabel("Y")
#ax.set_xlim([0.0, 1.0])
ax.set_ylim([min(min(list_pack), min(list_radius)), max(max(list_pack), max(list_radius))])
plt.legend()
plt.show()
print(list_radius)

print(points.values())
print(R)
mbh_time = time.process_time()
print(tech_time, " ", mbh_time - tech_time)
