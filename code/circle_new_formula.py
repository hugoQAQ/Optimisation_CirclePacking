import pyomo.environ as pe
import import_ipynb
from pyomo.core.base.block import generate_cuid_names
import random
import time
from optmodel_utilities import *
#from BoxConstrainedGO_Algorithms import MBH, multistart, purerandomsearch
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import decimal
from math import pi#, acos, asin#, cos, sin

# Define local path and function for loading solvers
path = "C:/Users/Pi-He/OneDrive/Bureau/Cours/Optimization/PyomoSolvers_win/solvers/"


def create_solver(solver_name="clpex"):
    #print(solver_name)
    solver_path = get_solver_path(solver_name)
    #print(solver_path)
    return  SolverFactory(solver_name, executable=str(solver_path), solver_io = 'nl')


# Define algorithms for Circle packing problem

# random generating point keeping in [lb,ub]
def random_point(model, gen_multi):
    for i in model.N:
        model.x[i] = gen_multi.uniform(model.lb, model.ub)
        model.y[i] = gen_multi.uniform(model.lb, model.ub)
        model.r = gen_multi.uniform(0.0, 1)

# perturbation
def perturb_point(model, gen_pert, delta):
    model.c= gen_multi.randint(1, model.n)
    print(model.c.value)
    model.p.value = model.n.value - model.c.value
    model.C = pe.RangeSet(1, model.c)
    model.P = pe.RangeSet(model.c + 1, model.n)
    print(model.P.value)
    print(model.C.value)
    print("C value ", model.c.value)

    for i in model.N:
        model.x[i] = model.x[i].value*(1+gen_pert.uniform(-1, 1) * delta)
        model.x[i] = max(model.lb, min(model.x[i].value, model.ub))
        model.y[i] = model.y[i].value * (1 + gen_pert.uniform(-1, 1) * delta)
        model.y[i] = max(model.lb, min(model.y[i].value, model.ub))


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
    #quit()
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
    model.c = pe.Param(default=int(round(size/2,0)), mutable=True)
    model.p = pe.Param(default=model.n-model.c, mutable=True)
    model.r_inf=0.3

    model.R_overlap=1/(size*pi)**(1/2)

    # set of variables, useful for sum and iterations
    model.N = pe.RangeSet(model.n)
    model.C= pe.RangeSet(1, model.c)
    model.P = pe.RangeSet(model.c+1, model.n)

    model.x = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.y = pe.Var(model.N,  bounds=(model.lb, model.ub))
    model.R = pe.Var(bounds=(0, 1))
    model.r = pe.Var(model.N, bounds=(0, 1))
    model.theta=pe.Var(model.N, bounds=(0.0, pi/2))


    def no_overlap_cartesian_rule(model, i , j):
        if i in model.C and j in model.C:
            if i < j:
                return(
                    (model.x[i] - model.x[j])**2
                    + (model.y[i] - model.y[j])**2 >= 4*model.R**2
                )
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip

    def no_overlap_polar_rule(model, i , j):
        if i in model.P and j in model.P:
            if i < j:
                return(
                model.r[i]**2 + model.r[j]**2-2*model.r[i]*model.r[j]*cos(model.theta[i]-model.theta[j])>=4*model.R**2
                )
            else:
                return pe.Constraint.Skip
        else:
            return pe.Constraint.Skip

    def no_overlap_reconciliation_rule(model, i, j):
        if i in model.P and j in model.C:
            return(
                    (model.x[i] - model.r[j]*cos(model.theta[j]))**2
                    + (model.y[i] - model.r[j]*sin(model.theta[j]))**2 >= 4*model.R**2
                )
        else:
            return pe.Constraint.Skip

    model.no_overlap_cartesian = pe.Constraint(model.C, model.C, rule=no_overlap_cartesian_rule)
    model.no_overlap_polar = pe.Constraint(model.P, model.P, rule=no_overlap_polar_rule)
    model.no_overlap_reconciliation = pe.Constraint(model.P, model.C, rule=no_overlap_reconciliation_rule)

    def r_rule(model, i):
        if i in model.P:
            return model.r[i] <= 1-model.R
        else:
            return pe.Constraint.Skip

    def xy_rule(model, i):
        if i in model.C:
            return model.x[i]**2+model.y[i]**2 <= (1-model.R)**2
        else:
            return pe.Constraint.Skip

    model.r_rule = pe.Constraint(model.N, rule=r_rule)
    model.xy_rule = pe.Constraint(model.N, rule=xy_rule)

    def Inside_x_min_rule(model, i):
        return model.x[i] >= model.R#model.r_inf+model.R
    model.Inside_x_min = pe.Constraint(model.C, rule=Inside_x_min_rule)

    def Inside_y_min_rule(model, i):
        return model.y[i] >= model.R#model.r_inf+model.R
    model.Inside_y_min = pe.Constraint(model.C, rule=Inside_y_min_rule)

    def Inside_x_max_rule(model, i):
        return model.x[i] <= 1-model.R
    model.Inside_x_max = pe.Constraint(model.C, rule=Inside_x_max_rule)

    def Inside_y_max_rule(model, i):
        return model.y[i] <= 1-model.R
    model.Inside_y_max = pe.Constraint(model.C, rule=Inside_y_max_rule)

    def radius_rule(model):
        return model.R

    def cos_rule(model, i):
        return model.x[i]==model.r[i]*cos(model.theta[i])

    def sin_rule(model, i):
        return model.y[i]==model.r[i]*sin(model.theta[i])

    model.cos_rule = pe.Constraint(model.N, rule=cos_rule)
    model.sin_rule = pe.Constraint(model.N, rule=sin_rule)

    def polar_r_rule(model, i):
        return model.r[i]>=model.r_inf+model.R

    def polar_theta_min_rule(model, i):
        return model.theta[i]>=asin(model.R/model.r[i])

    def polar_theta_max_rule(model, i):
        return model.theta[i]<=acos(model.R/model.r[i])

    model.polar_r = pe.Constraint(model.N, rule=polar_r_rule)
    model.polar_theta_min = pe.Constraint(model.N, rule=polar_theta_min_rule)
    model.polar_theta_max = pe.Constraint(model.N, rule=polar_theta_max_rule)


    # then we created the objective: function and sense of optimization
    model.obj = pe.Objective(rule=radius_rule, sense=pe.maximize)

    model.n = size

    # return instance
    return model.create_instance()



# ### 5.2. Model
# Define the model as circle packing with two points

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
'50':0.071377103864521467530864106392,
'51':0.071043138115537018210695844271,
'52':0.070957693072547251836538973586,
'53':0.069947252562026650314667369460,
'54':0.068645540104688887910867652499,
'55':0.068055360558639717449689169779,
'56':0.067532596322265705546305516123,
'57':0.067004873106043976620256140932,
'58':0.066232983745862232302511874085,
'59':0.065807496903601046306129157869,
'60':0.065030412648295594469592436244,
'61':0.064666268906273511710193139988,
'62':0.064252183294482911181458806768,
'63':0.064011528204864918571082299195,
'64':0.063458986813059169965842890472,
'65':0.063203957071856799974346494473,
'66':0.062862256903173650552663945791,
'67':0.062587429542203145161916665184,
'68':0.062520077997967509748911370819,
'69':0.061383571685582519393723042252,
'70':0.060596693631158157824224372564,
'71':0.060096531351830846614976309881,
'72':0.059801002126807327595495341667,
'73':0.059366050583080470834219283052,
'74':0.059082376336698616287994709302,
'75':0.058494535281249486795264924005,
'76':0.058198524936106167091962265813,
'77':0.057852577916326407945864081743,
'78':0.057702476734813469658915612215,
'79':0.057508497795177150643081806966,
'80':0.057370684146686868321824625645,
'81':0.056869921111948347401531481988,
'82':0.056512271650043116731600865688,
'83':0.056129373021650586640744529302,
'84':0.055856665888395536647508143405,
'85':0.055680181768308628605760305535,
'86':0.055572999412015161187514147010,
'87':0.054695259720704525732612598175,
'88':0.054406636912959447376712741080,
'89':0.053947040858284291446539903179,
'90':0.053749948306745947917965570800,
'91':0.053496719884488436041701634807,
'92':0.053317085175963914631383351067,
'93':0.052926433388240551617979070855,
'94':0.052795362726845093818338316930,
'95':0.052420366495943589361713667424,
'96':0.052275425469925719068554307283,
'97':0.052114795502244668305008905267,
'98':0.052032987702942098030211872968,
'99':0.051978606449505864308001821124,
'100':0.051401071774381815590184511455}


## Define variables
#algo = "MBH"
l_n = 20
u_n = 50
step=30
epsilon=10**-8
eps=8
delta=0.3
factor_no_improve=10

list_epsilon=[5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

print(list(range(l_n, u_n+1, 5)))

for algo in ['MBH']: #model.p = model.n-model.c
    for solver in ['knitro', 'snopt']:
        a = 0
        b = 1
        list_radius = []
        list_pack = []
        list_diff = []
        list_time = []
        list_ecart = []

        for i in range(l_n, u_n+1, step):#range(l_n, u_n+1):
            a+=1
            print("%s Circles"%i)
            init_time = time.process_time()

            mymodel = CirclePacking(i, 0, 1)

            Multi_n_iter = i * 5
            PRS_n_iter = Multi_n_iter * i * 20
            seed1 = 42

            gen_multi = random.Random()
            print(gen_multi)

            # chosing the solver
            localsolver = create_solver(path + solver)

            print(mymodel)

            # needed to retrieve variable names
            labels = generate_cuid_names(mymodel)

            logfile = open("mylog.txt", 'w')

            gen_multi.seed(seed1)

            if algo=="Multistart":
                FoundSolution, points = multistart(mymodel, Multi_n_iter, gen_multi, localsolver, labels, epsilon=10**-eps, logfile=logfile)

            elif algo == "MBH":
                FoundSolution, points = MBH(mymodel, gen_multi, localsolver, labels, max_no_improve=i * factor_no_improve, pert=random,
                                            epsilon=10**-eps, delta=delta, logfile=logfile)
            elif algo=="MBH_Multitrial":
                FoundSolution, points= MBH_MultiTrial(mymodel, 100, gen_multi, localsolver, labels, max_no_improve=i*factor_no_improve,
                                                      pert=random, epsilon=10**-eps, delta=delta, logfile=logfile)

            print(FoundSolution)
            tech_time = time.process_time()

            print(points)
            print(points.keys())

            X=[]
            Y=[]
            for j in points.keys():
                if 'x' in j:
                    X.append(points.get(j))
                elif 'y' in j:
                    Y.append(points.get(j))

            R=points.get('R')
            print(R)
            list_radius.append(R)
            list_pack.append(Packomania_radius.get(str(i)))

            list_ecart.append(R-Packomania_radius.get(str(i)))
            list_diff.append((R/Packomania_radius.get(str(i))-1)*100)
            list_time.append(tech_time-init_time)

            # print(points.get(''))
            fig, ax = plt.subplots()
            import matplotlib.colors

            for k in range(len(X)):
                plt.plot(X[k], Y[k], marker='+')
                circle = plt.Circle((X[k], Y[k]), R, color='gold')
                ax.add_artist(circle)

            circle = plt.Circle((0, 0), 1, fill=False)
            ax.add_artist(circle)

            circle = plt.Circle((0, 0), mymodel.r_inf, fill=False)
            ax.add_artist(circle)

            plt.title("N=%s (R=%s)" % (i, R))
            plt.xlabel("X")
            plt.ylabel("Y")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            plt.show()



        pd.options.display.float_format = "{:,.30f}".format
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        print(len(list_radius), len(list_pack), len(list_diff), len(range(l_n, u_n+1)))
        df=pd.DataFrame({'N':range(l_n, u_n+1, step), 'Radius':list_radius, 'Packomania':list_pack, 'Ecart absolu':list_ecart, 'Ecart [%]':list_diff, 'Time [s]':list_time})
        #df['Ecart [%]'] = df['Ecart [%]'].apply(lambda x: round(x, 2))
        #df['Time [s]'] = df['Time [s]'].apply(lambda x: round(x, 2))
        df['Ecart [%]']= df['Ecart [%]'].round(5)
        df['Time [s]']= df['Time [s]'].round(5)
        df['Radius']= df['Radius'].round(30)
        df['Packomania']= df['Packomania'].round(30)
        df['Packomania'] = df['Packomania'].apply(lambda x: round(x, 30))

        print(df)

        X = []
        Y = []
        for i in points.keys():
            if 'x' in i:
                X.append(points.get(i))
            elif 'y' in i:
                Y.append(points.get(i))

        # x1=points.get('x:#1')
        # x2=points.get('x:#2')
        # y1=points.get('y:#1')
        # y2=points.get('y:#2')
        R = points.get('r')

        print(points.values())
        print(R)
        mbh_time = time.process_time()
        print(tech_time, " ", mbh_time - tech_time)

        with open('Result_Lopez_Beasley_'+str(algo)+'_'+str(solver)+'_eps10-'+str(eps)+'_delta'+str(round(delta,2))+'.txt', 'w') as f:
            print(df, file=f)

        fig, ax = plt.subplots()
        absc=list(range(l_n, u_n+1, step))
        plt.plot(absc, list_radius, label=algo)
        plt.plot(absc, list_pack, label="Packomania")
        plt.title("R=f(N)")
        ax.set_ylim([min(min(list_pack), min(list_radius)), max(max(list_pack), max(list_radius))])
        plt.legend()
        plt.savefig('Result_Lopez_Beasley_'+str(algo)+'_'+str(solver)+'_eps10-'+str(eps)+'_delta'+str(round(delta,2))+'.png')
        #plt.show()
        print(list_radius)