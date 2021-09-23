# %%
# conda install git pip
# pip install qutip
# %%

import numpy as np
import qutip as qt
from qutip import ket2dm as dm
from qutip import basis as B
# from qutip import entropy_vn as vn
# %% DEFINING STATES

# Defining the coefficients and errors
def r(v):
    return (1+3*v)/4, (1-v)/4, (1-v)/4, (1-v)/4

def e_N(v,N):
    return ((1-v)**N)/((1-v)**N+(1+v)**N)

# AB post-CAD projectors
def AB(i,j):
    return dm(qt.tensor(B(2,i), B(2,j)))

# E post AB measurement states post-CAD
def E_N(i,j,v,N):
    if i==j:
       e_ij= (np.sqrt(r(v)[0])*B(4,0) + (-1)**i*np.sqrt(r(v)[1])*B(4,1))/(np.sqrt(r(v)[0]+r(v)[1]))
       return qt.tensor([dm(e_ij) for i in range(N)])
    else:
        e_ij= (np.sqrt(r(v)[2])*B(4,2) + (-1)**i*np.sqrt(r(v)[3])*B(4,3))/(np.sqrt(r(v)[2]+r(v)[3]))
        return qt.tensor([dm(e_ij) for i in range(N)])

# Defining the CCQ states
def ABE_state(v,N):
    err = e_N(v,N)
    nerr = 1 - err
    eq = qt.tensor(AB(0,0), (E_N(0,0,v,N))) + qt.tensor(AB(1,1), E_N(1,1,v,N))
    dif = qt.tensor(AB(0,1), (E_N(0,1,v,N))) + qt.tensor(AB(1,0), E_N(1,0,v,N))
    return nerr/2*eq + err/2*dif

# This function calculates the trace of each subsystem
def partial_ABE(X, N, i):
    if i==0 or i==1:
        return X.ptrace(i)
    elif i==2:
        return qt.tensor([X.ptrace(i) for i in range(2, N+2)])
    else:
        print('Please input i=0 for Alice, i=1 for Bob or i=2 for Eve')


# %% TESTING SOME STUFF
stat_func = ABE_state(1/2,1)
err = e_N(1/2,1)
stat = (1-err)/2*(qt.tensor(AB(0,0), E_N(0,0,1/2,1)) + qt.tensor(AB(1,1), E_N(1,1,1/2,1))) + err/2*(qt.tensor(AB(0,1), E_N(0,1,1/2,1)) + qt.tensor(AB(1,0), E_N(1,0,1/2,1)))
stat==stat_func 
# %% TESTING SOME MORE STUFF
qt.tensor(E_N(0,0,1/2,1), E_N(0,0,1/2,1))==qt.tensor(list(E_N(0,0,1/2,1) for i in range(2)))
c = qt.tensor(E_N(0,0,1/2,1),E_N(0,0,1/2,1),E_N(0,0,1/2,1))
b = [E_N(0,0,1/2,1) for i in range(2)]
qt.tensor(E_N(0,0,1/2,1), qt.tensor(b)) == c

# %% TESTING EVEN MORE STUFF
E_test = qt.tensor(E_N(1,1,1/2,1), E_N(0,0,1/2,2), E_N(1,1,1/2,1))
partial_ABE(E_test,2,2)==qt.tensor(E_N(0,0,1/2,1), E_N(1,1,1/2,1))


# %% FINALLY WE COMPUTE I(A:B|E)

def cond_mut_inf(X, N):
    S_AE = qt.entropy_vn(qt.tensor(partial_ABE(X,N,0), partial_ABE(X,N,2)), base=2)
    S_BE = qt.entropy_vn(qt.tensor(partial_ABE(X,N,1), partial_ABE(X,N,2)), base=2)
    S_ABE = qt.entropy_vn(X, base=2)
    S_E = qt.entropy_vn(partial_ABE(X,N,2), base=2)
    return S_AE+S_BE-S_ABE-S_E

def master_bound(v,N):
    return cond_mut_inf(ABE_state(v,N), N)


# %% THE LAST BIT OF TESTING BEFORE SDP
master_bound(1,1)
E_N(1,1,1,1)==E_N(0,0,1,1)
(1/2*qt.tensor(AB(0,0), E_N(0,0,1,1))+1/2*qt.tensor(AB(1,1), E_N(1,1,1,1)))==ABE_state(1,1)
qt.entropy_vn(qt.Qobj(np.eye(2,2)/2), base=2)
master_bound(1/2,1)

# %% FIGURING OUT SUPEROPERATORS HERE
zz = dm(qt.tensor(B(2,0),B(2,0)))
sx = qt.tensor(qt.sigmax(), qt.sigmax())
Sx = qt.to_super(sx)
Sx*qt.operator_to_vector(zz)
sx, sx.dag()
A = qt.Qobj(np.arange(4).reshape(2,2))
A, A.dag()

# Essentially to operate with a channel on a density matrix, I either do x*rho*xdag
# or i turn x into a superoperator X=qt.to_super(x) and then i do 
# X*qt.operator_to_vector(rho)


# %% Testing some thangs
z=qt.sigmax()
U = qt.Qobj(np.exp(1j*z.full()))
U*U.dag()

