#%%
import numpy as np
import matplotlib.pyplot as plt
# %% Defining the binary entroup and conditional informations post cad
def h(x):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)

def e_AB(v):
    return (1-v)/2
def e_N(v, N):
    return (1-v)**N/((1-v)**N+(1+v)**N)

# def cond_inf(v,N):
#     return (1-((1-v)**N)/((1-v)**N+(1+v)**N))*(1-h((1-(2*v/(1+v))**N)/2))

def cond_inf(v,N):
    return (1-e_N(v,N))*(1-h((1-(2*v/(1+v))**N)/2))

def mut_inf(v, N):
    return (1-e_N(v,N))*(1-h((1-(2*v/(1+v))**N)/2))-h(e_N(v, N))

h(1/2)
cond_inf(1,1)
x = np.sqrt(5)/5
# %%
plt.figure('One')
v=np.linspace(0,1,500)
CI_10 = cond_inf(v, 10)
CI_5 = cond_inf(v, 5)
CI_1 = cond_inf(v, 1)
MI_10 = mut_inf(v,10)
MI_5 = mut_inf(v,5)
MI_1 = mut_inf(v,1)
plt.plot(v, CI_1, color='red', label='Cond_Inf for N=1')
plt.plot(v, CI_5, color='blue', label='Cond_Inf for N=5')
plt.plot(v, CI_10, color='green', label='Cond_Inf for N=10')
plt.plot(v, MI_1, color='yellow', label='Mut_Inf for N=1')
plt.plot(v, MI_5, color='purple', label='Mut_Inf for N=5')
plt.plot(v, MI_10, color='black', label='Mut_Inf for N=10')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('e_N')
plt.title('Post-Cad info bounds')
plt.show()

# %% CONDITIONAL MUTUAL INFO POST CAD I(A:B|E)(v)

plt.figure('Two')
x = np.linspace(1/3,1,500)
CI_1 = cond_inf(x, 1)
CI_5 = cond_inf(x, 5)
CI_10 = cond_inf(x, 10)
CI_20 = cond_inf(x, 20)

plt.plot(x, CI_1, color='red', label='N=1')
plt.plot(x, CI_5, color='blue', label='N=5')
plt.plot(x, CI_10, color='green', label='N=10')
plt.plot(x, CI_20, color='black', label='N=20')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('Info'), plt.title('Post-CAD conditional informations as function of v')
plt.show()
plt.close()

# %% CONDITIONAL MUTUAL INFO POST CAD I(A:B|E)(N)


plt.figure('Three')
N = np.linspace(1,100,500)
CIN_3 = cond_inf(1/3, N)
CIN_2 = cond_inf(0.5, N)
CIN_23 = cond_inf(2/3, N)
CIN_1 = cond_inf(1, N)

plt.plot(N, CIN_3, color='red', label='v=1/3')
plt.plot(N, CIN_2, color='blue', label='v=1/2')
plt.plot(N, CIN_23, color='green', label='v=2/3')
plt.plot(N, CIN_1, color='black', label='v=1')
plt.legend()
plt.xlabel('N'), plt.ylabel('Info')
plt.title('Post-CAD conditional informations as function of N')
plt.show()
plt.close()

# %% WIDTH OF KEY DISTILLATION
plt.figure('Four')
v = np.linspace(1/3,1,500)
he_AB = h(e_AB(v))
he_N = h(e_N(v,N=5))
plt.plot(v, he_AB, color='red', label='pre-CAD N=1')
plt.plot(v, he_N, color='blue', label='post-CAD N=5')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('Binary entropy'), plt.title('Width of conditional informations')
plt.show()
plt.close()

# %% CHECKING OUT eN:

plt.figure('Five_1')
v=np.linspace(0,1,500)
e_10 = e_N(v, 10)
e_5 = e_N(v, 5)
e_1 = e_N(v, 1)
plt.plot(v, e_1, color='red', label='e_N(v) for N=1')
plt.plot(v, e_5, color='blue', label='e_N(v) for N=5')
plt.plot(v, e_10, color='green', label='e_N(v) for N=10')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('e_N'), plt.title('Post-CAD error as function of v')
plt.show()

plt.figure('Five_2')
N=np.linspace(1,20,500)
e_N_1 = e_N(1/3, N)
e_N_2 = e_N(0.5, N)
e_N_3 = e_N(2/3, N)
plt.plot(N, e_N_1, color='red', label='e_N(N) for v=1/3')
plt.plot(N, e_N_2, color='blue', label='e_N(N) for v=1/2')
plt.plot(N, e_N_3, color='green', label='e_N(N) for v=2/3')
plt.legend()
plt.xlabel('N'), plt.ylabel('e_N'), plt.title('Post-CAD error as function of N')
plt.show()
plt.close()


# %% CHECKING OUT THE RANGE v \in (1/3, sqrt(5)/5]
plt.figure('Six')
v=np.linspace(1/3,np.sqrt(5)/5,500)
CI_10 = cond_inf(v, 10)
CI_5 = cond_inf(v, 5)
CI_1 = cond_inf(v, 1)
plt.plot(v, CI_1, color='red', label='Cond_Inf for N=1')
plt.plot(v, CI_5, color='blue', label='Cond_Inf for N=5')
plt.plot(v, CI_10, color='green', label='Cond_Inf for N=10')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('e_N'), plt.title('Post-Cad info in dark range')
plt.show()
# %% CHECKING OUT eN IN DARK RANGE FOR POST POVM UPPER BOUND:

plt.figure('Seven_1')
v=np.linspace(1/3,np.sqrt(5)/5,500)
e_10 = e_N(v, 10)
e_5 = e_N(v, 5)
e_1 = e_N(v, 1)
plt.plot(v, e_1, color='red', label='e_N(v) for N=1')
plt.plot(v, e_5, color='blue', label='e_N(v) for N=5')
plt.plot(v, e_10, color='green', label='e_N(v) for N=10')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('e_N'), plt.title('Post Eve POVM Upper bound as func of v')
plt.show()

plt.figure('Seven_2')
N=np.linspace(1,20,500)
e_N_1 = e_N(1/3, N)
e_N_2 = e_N((1/3+np.sqrt(5)/5)/2, N)
e_N_3 = e_N(2/3, N)
plt.plot(N, e_N_1, color='red', label='e_N(N) for v=1/3')
plt.plot(N, e_N_2, color='blue', label='e_N(N) for v=midway')
plt.plot(N, e_N_3, color='green', label='e_N(N) for v=sqrt(5)/5')
plt.legend()
plt.xlabel('N'), plt.ylabel('e_N'), plt.title('Post Eve POVM Upper bound as func of N')
plt.show()
plt.close()

# %% CHECKING OUT eN-h(eN) IN DARK RANGE FOR POST POVM LOWER BOUND:

plt.figure('Eight')
v=np.linspace(1/3,np.sqrt(5)/5,500)
LB_1 = e_N(v, 1)-h(e_N(v, 1))
LB_2 = e_N(v, 5)-h(e_N(v, 5))
LB_3 = e_N(v, 10)-h(e_N(v, 10))
plt.plot(v, LB_1, color='red', label='e_N-h(e_N) for N=1')
plt.plot(v, LB_2, color='blue', label='e_N-h(e_N) for N=5')
plt.plot(v, LB_3, color='green', label='e_N-h(e_N) for N=10')
plt.legend()
plt.xlabel('Visibility'), plt.ylabel('Diff Mutual Infos'), plt.title('Post Eve POVM Lower bound as func of v')
plt.show()
plt.close()


# %% LOWER BOUND WITH INDIVIDUAL POVMS
def lam_eq(v,N):
    return (2*v/(1+v))**N

def mut_dif(v,N):
    return e_N(v,N)-h(e_N(v,N))+(1-e_N(v,N))*lam_eq(v,N) 

def mut_dif2(v,N):
    return e_N(v,N)*(1+np.log2(e_N(v,N)))+(1-e_N(v,N))*(np.log2(1-e_N(v,N))+lam_eq(v,N))

plt.figure('Nine')
v = np.linspace(1/3,x,500)
LB_1 = mut_dif(v,1)
LB_2 = mut_dif(v,5)
LB_3 = mut_dif(v,10)
LB_4 = mut_dif(v,100)
plt.plot(v, LB_1, color='red', label='e_N-h(e_N)+(1-e_N)Lam_eq^N for N=1')
plt.plot(v, LB_2, color='blue', label='e_N-h(e_N)+(1-e_N)Lam_eq^N for N=5')
plt.plot(v, LB_3, color='green', label='e_N-h(e_N)+(1-e_N)Lam_eq^N for N=10')
plt.plot(v, LB_4, color='purple', label='e_N-h(e_N)+(1-e_N)Lam_eq^N for N=100')

plt.legend()
plt.xlabel('Visibility'), plt.ylabel('Diff Mutual Infos'), plt.title('Post Eve POVM Lower bound as func of v')
plt.show()
plt.close()
mask = LB_4>=0
v[mask]


# %% SAME AS ABOVE BUT WITH UPPER BOUND

def cont_mut(v,N):
    e = e_N(v,N)
    l = lam_eq(v,N)
    return (1-e)*l+e-((1-e)*l+e)*np.log2((1-e)*l+e)+(1-e)*l*np.log2((1-e)*l)+e*np.log2(e)

plt.figure('Ten')
v = np.linspace(1/3,x,500)
UB_1 = cont_mut(v,1)
UB_2 = cont_mut(v,5)
UB_3 = cont_mut(v,10)
UB_4 = cont_mut(v,100)
plt.plot(v, UB_1, color='red', label='I(A:B|U) for N=1')
plt.plot(v, UB_2, color='blue', label='I(A:B|U) for N=5')
plt.plot(v, UB_3, color='green', label='I(A:B|U) for N=10')
plt.plot(v, UB_4, color='purple', label='I(A:B|U) for N=100')

plt.legend()
plt.xlabel('Visibility'), plt.ylabel('Cond Mutual Info'), plt.title('Post Eve POVM Upper bound as func of v')
plt.show()
plt.close()
UB_2>=LB_2