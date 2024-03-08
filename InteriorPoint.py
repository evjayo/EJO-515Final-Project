# this file contains collection of solver we learned in the class
from numpy.linalg import norm
from numpy.linalg import solve
import numpy as np

#Interior Point Methods for Linear Functions
# -----------------------------------------------------------------------------
def lsInteriorPoint(x0, A, b, C, d, tol=1e-10, max_iter=1000):
    """
    Optimize Least Squares with Interior Point Methods
    
    input
    -----
    x0  : array_like
        Starting point.
    A   : array_like
        matrix
    b   : array_like
        vector
    C   : array_like
        constraint matrix
    d   : array_like
        constraint vector
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x   : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    pt_his  : array_like
        Point history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    #Initialize Variables
    x = np.copy(x0)
    obj = np.inner((A@x - b),(A@x - b))/2
    m = d.size
    n = x.size
    z = np.ones(m)
    s = np.ones(m)
    mu = 1
    #more initialization
    obj_his = np.zeros(max_iter + 1)
    pt_his = []
    #even more
    obj_his[0] = obj
    pt_his.append([np.copy(x)])
    cond = 1!=0
    
    #start the iteration
    iter_count = 0
    while cond:
        S = np.diag(s)
        Z = np.diag(z)
        Fh = np.block([[np.eye(m,m),np.zeros((m,m)),C],[Z,S,np.zeros((m,n))],[np.zeros((n,m)),C.T,A.T@A]])
        Fg = np.block([s + C@x - d,S@z - mu*np.ones(m),A.T@(A@x - b)])
        u = solve(Fh,-Fg)
        su = u[0:m]
        zu = u[m:m+m]
        xu = u[m+m:m+m+n]
        a = 1
        while any(z + a*zu <= 0) or any(s + a*su <= 0):
            a *= .9
        x += a*xu
        s += a*su
        z += a*zu
        mu = np.inner(s,z)/(10*m)
        iter_count += 1
        obj = np.inner((A@x - b),(A@x - b))/2
        obj_his[iter_count] = obj
        pt_his.append([np.copy(x)])
        err = abs(obj_his[iter_count]-obj_his[iter_count - 1])
        g = norm(A.T@(A@x - b))
        cond = g > tol and a > tol
        if iter_count >= max_iter:
            print('Interior Point Method reached the maximum number of iteration.')
            return x, obj_his[:iter_count+1], pt_his, 1
    return x, obj_his[:iter_count+1], pt_his, 0
        
def interiorPoint(x0, func, grad, hess, C, d, tol=1e-6, max_iter=1000):
    """
    Optimize a general convex function with interior point methods on a polytope
    
    input
    -----
    x0  : array_like
        Starting point
    func: function
        objective function
    grad: function
        gradient of objective function
    hess: function
        hessian of objective function
    C   : array_like
        constraint matrix
    d   : array_like
        constraint vector
    tol : float, optional
        Gradient tolerance for termination of solver
    max_iter : int, optional
        Maximum number of iterations
    
    output
    ------
    x   : array_like
        final solution
    pt_his : array_like
        history of the interior point
    obj_his : array_like
        history of the objective function
    err_his : array_like
        norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """   
    #Initialize Variables
    x = np.copy(x0)
    g = grad(x)
    H = hess(x)
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    pt_his = []
    #
    obj_his[0] = obj
    err_his[0] = err
    pt_his.append([np.copy(x)])
    #
    n = x.size
    m = d.size
    #
    z = np.ones(m)
    s = np.ones(m)
    mu = 1
    cond = 1!=0
    
    #start the iteration
    iter_count = 0
    while cond:
        S = np.diag(s)
        Z = np.diag(z)
        Fh = np.block([[np.eye(m,m),np.zeros((m,m)),C],[Z,S,np.zeros((m,n))],[np.zeros((n,m)),C.T,H]])
        Fg = np.block([s + C@x - d,S@z - mu*np.ones(m),g + C.T@z])
        u = solve(Fh,-Fg)
        su = u[0:m]
        zu = u[m:2*m]
        xu = u[2*m:2*m+n]
        a = 1#/max(abs(np.linalg.eigvals(H)))
        while (any(s + a*su <= 0) or any(z + zu*a <= 0) ):#or func(x+a*xu)>=func(x)) #and a >= 1e-6:
            a *= .9
        x += a*xu
        s += a*su
        z += a*zu
        mu = np.inner(s,z)/(10*m)
        g = grad(x)
        H = hess(x)
        obj = func(x)
        err = norm(g)
        iter_count +=1
        err_his[iter_count] = err
        obj_his[iter_count] = obj
        pt_his.append([np.copy(x)])
        cond = a > tol and err > tol
        if iter_count >= max_iter:
            print('Interior Point Method reached the maximum number of iteration.')
            return x, obj_his, err_his, pt_his, 1
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], pt_his, 0

def constrainedGD(x0, func, grad, C, d, step_size, tol=1e-10, max_iter=1000):
    """
    input
    -----
    x0 : array_like
        Starting point for the solver.
    func : function
        Input x and return the function value.
    grad : function
        Input x and return the gradient.
    C    : array_like
        Constraint matrix
    d    : array_like
        Constraint vector
    step_size : float
        beta smoothness constant
    tol : float, optional
        Gradient tolerance for terminating the solver.
    max_iter : int, optional
        Maximum number of iteration for terminating the solver.
        
    output
    ------
    x : array_like
        Final solution
    obj_his : array_like
        Objective function value convergence history
    err_his : array_like
        Norm of gradient convergence history
    exit_flag : int
        0, norm of gradient below `tol`
        1, exceed maximum number of iteration
        2, others
    """
    # initial information
    x = np.copy(x0)
    g = grad(x)
    #
    obj = func(x)
    err = norm(g)
    #
    obj_his = np.zeros(max_iter + 1)
    err_his = np.zeros(max_iter + 1)
    #
    obj_his[0] = obj
    err_his[0] = err
    cond = 0!=1
    
    # start iterations
    iter_count = 0
    while cond:
        # gradient descent step
        s = step_size
        while any(C@(x - s*g) > d):
            s *= .9
        x -= s*g
        #
        # update function and gradient
        g = grad(x)
        #
        obj = func(x)
        err = norm(g)
        #
        iter_count += 1
        obj_his[iter_count] = obj
        err_his[iter_count] = err
        cond = s > tol and err > tol
        #
        # check if exceed maximum number of iteration
        if iter_count >= max_iter:
            print('Gradient descent reach maximum number of iteration.')
            return x, obj_his[:iter_count+1], err_his[:iter_count+1], 1
    #
    return x, obj_his[:iter_count+1], err_his[:iter_count+1], 0