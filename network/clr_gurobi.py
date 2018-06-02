# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

from gurobipy import *
import gurobipy
import numpy as np
#import time


class CLR(object):
    def __init__(self, nB, nK):
        self.nB = nB
        self.nK = nK

        try:
            # Create a new model.
            self.m = Model("clr")

            # Disable solver output.
            self.m.setParam('OutputFlag', 0)

            # Add variables and constraints.
            self.var_X = [[None] * self.nK for k in range(self.nB)]
            for k in range(self.nB):
                for i in range(self.nK):
                    self.var_X[k][i] = self.m.addVar(vtype=GRB.CONTINUOUS,
                            name='x_{:d}_{:d}'.format(k,i))

            # Create variable lists for quadratic terms.
            self.var_X_quads_1 = [[] for k in range(self.nB)]
            self.var_X_quads_2 = [[] for k in range(self.nB)]
            for k in range(self.nB):
                for i in range(self.nK):
                    for j in range(self.nK):
                        self.var_X_quads_1[k].append(self.var_X[k][i])
                        self.var_X_quads_2[k].append(self.var_X[k][j])

            # NOTE:
            # http://www.gurobi.com/support/faqs
            # For best performance, you should create all variables first,
            # then add constraints.
            for k in range(self.nB):
                for i in range(self.nK):
                    self.m.addConstr(self.var_X[k][i] >= 0,
                            name='c1_{:d}_{:d}'.format(k,i))
                    self.m.addConstr(self.var_X[k][i] <= 1,
                            name='c2_{:d}_{:d}'.format(k,i))

            self.m.update()

        except GurobiError as e:
            print('[init] Encountered a Gurobi error: {}'.format(e))
            sys.exit(-1)
        except AttributeError as e:
            print('[init] Encountered an attribute error: {}'.format(e))
            sys.exit(-1)


    def solve(self, AtA, btA, AtA_diag=0.):
        # argmin_x || Ax - b ||_2^2 s.t. 0 <= x <= 1
        # argmin_x (x^T A^T A x - 2 b^T A x) s.t. 0 <= x <= 1
        # AtA: (B, K, K)
        # btA: (B, K)
        # x: (B, K)

        assert(AtA.shape[0] == self.nB)
        assert(AtA.shape[1] == self.nK)
        assert(AtA.shape[2] == self.nK)
        assert(btA.shape[0] == self.nB)
        assert(btA.shape[1] == self.nK)

        try:
            # Set objective.
            objective = 0
            for k in range(self.nB):
                # NOTE:
                # http://www.gurobi.com/documentation/8.0/refman/py_quadexpr.html
                # The most efficient way to build a large quadratic expression
                # is with a single call to addTerms.
                AtA_quads = []
                for i in range(self.nK):
                    for j in range(self.nK):
                        if i == j and AtA_diag > 0.:
                            # NOTE:
                            # Add a regularizing term to the diagonal.
                            AtA_quads.append(AtA[k,i,j] + AtA_diag)
                        else:
                            AtA_quads.append(AtA[k,i,j])

                # a_ij * x_i * x_j
                expr = QuadExpr()
                expr.addTerms(AtA_quads,
                        self.var_X_quads_1[k], self.var_X_quads_2[k])

                # -2 * b_i * x_i
                expr.addTerms((-2. * btA[k,:]).tolist(), self.var_X[k])
                objective += expr

            self.m.setObjective(objective, GRB.MINIMIZE)

            # Optimize.
            #tic = time.clock()
            self.m.optimize()
            assert (self.m.Status == GRB.OPTIMAL)
            #toc = time.clock()
            #print('Elapsed time: {}'.format(toc - tic))

            x = np.empty((self.nB, self.nK))
            for k in range(self.nB):
                for i in range(self.nK):
                    x[k,i] = self.var_X[k][i].x

        except GurobiError as e:
            if AtA_diag == 0.:
                # Retry with a regularizer.
                return self.solve(AtA, btA, AtA_diag + 1.)
            else:
                print('[solve] Encountered a Gurobi error: {}'.format(e))
                sys.exit(-1)
        except AttributeError as e:
            print('[solve] Encountered an attribute error: {}'.format(e))
            sys.exit(-1)

        return x

