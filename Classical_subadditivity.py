# %%
import numpy as np

# %%
def H(l):
    return (l*np.log2(l)).sum()

def h(x):
    return -x*np.log2(x)-(1-x)*np.log2(1-x)
# %% Entropy joint
x = 4*4*1/48*np.log2(1/48)+2*(2*2*(1+1)*1/24*np.log2(1/24))
H_XYForg = -x
H_XYForg

y = 4*1/12*np.log2(1/12)+2*(4*1/12*np.log2(1/12))
H_XForg = -y

I_XYForg = 2*H_XForg-H_XYForg
I_XYForg
# %% Reduced intrinsic
RIInf = h(1/4)+3/4*I_XYForg
RIInf
# %% Just some calculations
HX = 4*1/4*np.log2(1/4)
HXY = np.log2(1/16)
2*HX-HXY

# Indeed, the mutual info is 0 for a uniform distn
