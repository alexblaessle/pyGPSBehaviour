"""Test script to understand integer programming in python."""

import numpy as np
import cvxpy

def constrainFunc(x):
	
	np.diff(x[::-1]).any()>0

#def objFunc(x):
	
	#return (x[0]-2)+x[1]+(x[2]-4)**2


x=cvxpy.Variable(3)

objFunc=cvxpy.power(x[0],2)-2

print objFunc
print type(objFunc)

#x = cvxpy.Int(*x.shape)

#contrain=np.diff(x[::-1]).any()>0

#problem = cvxpy.Problem(cvxpy.Minimize(objFunc), constraints=constrain)



objective = cvxpy.Minimize(objFunc(x))
problem = cvxpy.Problem(cvxpy.Minimize(objective),[])

problem.solve(solver=cvxpy.GLPK_MI)



