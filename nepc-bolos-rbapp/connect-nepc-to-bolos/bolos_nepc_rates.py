# %% [markdown] BOLOS file that formats nepc data as matrix rate file for RB app
# NOTE: Start with an initial grid and then expand to a quadratic one after adjusting the the mean energy of the first grid
# TODO: Check 'lion' commits on github and his manual
# TODO: Add in a closer model that uses the ideal gas law
# TODO: Check the data and units!
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
from matplotlib import rcParams
from collections import defaultdict
import nepc
from nepc.util import config
from bolos import parser, grid, solver

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['font.weight'] = 400

rcParams['mathtext.rm'] = 'serif'
rcParams['mathtext.it'] = 'serif:italic'
rcParams['mathtext.bf'] = 'serif:bold'
rcParams['mathtext.fontset'] = 'custom'

# %%
enefile = []
ratefile = []
swarmfile = []
if __name__ == '__main__':
    EoN = np.linspace(0.1,2000)
    press = 101325
    T_k = 300
    ND = press / co.k / T_k
    
    with open('Cross section.txt') as fp:
        processes = parser.parse(fp)

    cnx, cursor = nepc.connect(local=False)
    n_phelps = nepc.Model(cursor, "phelps")
    
    for en in EoN:
        g = grid.LinearGrid(0, 100, 500) # 0-100ev 500 intervals
        bsolver = solver.BoltzmannSolver(g)
        bsolver.load_collisions(processes)
        bsolver.target['N2'].density = 1
        bsolver.kT = T_k * co.k / co.eV
        bsolver.EN = en * solver.TOWNSEND       
        bsolver.init()
        f0 = bsolver.maxwell(2.0)
        f_sol = bsolver.converge(f0, maxn=200, rtol=1e-4)
        mean_energy = bsolver.mean_energy(f_sol)
        electron_temp = bsolver.electron_temperature(f_sol)     
        newgrid = grid.QuadraticGrid(0, 50 * mean_energy, 200)
        bsolver.grid = newgrid
        bsolver.init()

        finterp = bsolver.grid.interpolate(f_sol, g)
        f1 = bsolver.converge(finterp, maxn=200, rtol=1e-5)

        mu = bsolver.mobility(f1) / ND
        diff = bsolver.diffusion(f1) / ND
        k = bsolver.rate(f1, "N2 -> N2^+")

        enefile.append([en, mean_energy])
        ratefile.append([mean_energy,] + [bsolver.rate(f1, p) if bsolver.rate(f1, p) > 1.0e-25 else 0.0e00 for t, p in bsolver.iter_all()])
        swarmfile.append([en, mean_energy, electron_temp, mu, diff, k])

    with open('energy_v_Td_bolos_nepc.txt', 'w') as f:
        np.savetxt(f, np.c_[enefile], delimiter='         ', fmt='%0.3e') # the 0 is as many places as needed
    
    with open('bolos_nepc_rates.txt', 'w') as f:
        np.savetxt(f, np.r_['0,2', ratefile], delimiter='                   ', fmt='%0.3e') 
    
    with open('swarm_file_nepc.txt', 'w') as f:
        np.savetxt(f, np.c_[swarmfile], delimiter='         ', fmt='%0.3e')
# %%
[(p['kind'], p['target'], p['comment'], p.get('threshold'), p.get('product','')) for p in processes]
comment = [p['comment'] for p in processes]
kind = map(str.capitalize, [p['kind'] for p in processes])
reatctant = [p['target'] for p in processes]
product = [p.get('product','N2(X1)') for p in processes]
thresh = [p.get('threshold','0.0') for p in processes]

new_comment = [y for x in comment for y in x.split('\n')]
new_comment = [x for x in new_comment if not x.startswith('UPDATED')]
new_comment = [x for x in new_comment if not x.startswith('COLUMNS')]

# I have a couple options below which are used to extract data
# One is a dictionary of dictionaries, the other is a dictionary of lists
# Using the dictionary of lists right now
d = {}

for i in new_comment:
    #d.setdefault(i.split(': ')[0],set()).add(i.split(': ')[1])
    d.setdefault(i.split(': ')[0],[]).append(i.split(': ')[1].replace(' ',''))

# for n in new_comment:
#     k, v = n.split(': ')
#     if k in d:
#         d[k].add(v)
#         d[k].append(v)
#     else:
#         d[k] = {v}
#         d[k] = [v]

words = [',effective', ',excitation', ',Ionization', ',completeset']
for word in words:
    if word in words:
        d['PROCESS'] = [item.replace('E', 'e') for item in d['PROCESS']]
        d['PROCESS'] = [item.replace('N2+(B2SIGMA)', 'N2(X2:ion:B2SIGMA)') for item in d['PROCESS']]
        d['PROCESS'] = [item.replace('N2(r', 'N2(R') for item in d['PROCESS']]
        d['PROCESS'] = [item.replace('N2+', 'N2(X2:ion)') for item in d['PROCESS']]
        d['PROCESS'] = [item.replace('N2-', 'N2(X1)-') for item in d['PROCESS']]
        d['PROCESS'] = [item.replace(word, '') for item in d['PROCESS']]
        d['PARAM.'] = [item.replace(word, '') for item in d['PARAM.']]
    else:
        break

d['PROCESS'][0] = 'e+N2(X1)->e+N2(X1)'

th = f"{'!Thresh(eV)':<28}{''.join([format(str(x),'<28') for x in [*thresh]])}\n"
re = f"{'!Reaction':<28}{''.join([format(str(x),'<28') for x in d['PROCESS']])}\n"
ty = f"{'!Type':<28}{''.join([format(str(x),'<28') for x in [*kind]])}\n"
units = f"{'!<E> (eV)':<28}{''.join(format('Rate_Constant(m^3/s)','<28'))*26}\n"
sep = f"{'!'}{''.join('-')*len(re)}\n"

string = [th, re, ty, units, sep]

# %%
header1 = f"{'E/N (Td)':<18}{'Energy (eV)'}\n"
file1 = 'energy_v_Td_bolos_nepc.txt'
header2 = f"{''.join(string)}"
file2 = 'bolos_nepc_rates.txt'
header3 = f"{'E/N (Td)':<18}{'Energy (eV)':<18}{'Temp (K)':<18}{'Mobility ()':<20}{'Diffusion ()':<20}{'Ionization Rate ()':<20}\n"
file3 = 'swarm_file_nepc.txt'

def WriteHeader(filename, header):
    with open(filename, "r") as f:
        lines = f.readlines()
        f.close()
    lines.insert(0, header)
    with open(filename, "w") as f:
        lines = "".join(lines)  
        f.write(lines)
        f.close()

a = WriteHeader(file1, header1)
b = WriteHeader(file2, header2)
c = WriteHeader(file3, header3)
# %%
data_bolos = np.loadtxt('energy_v_Td_bolos_nepc.txt', skiprows=1)
data_bolsig = np.loadtxt('energy_v_Td_bolsig+.txt')

x1 = data_bolos[:, 0]
y1 = data_bolos[:, 1]
x2 = data_bolsig[:, 0]
y2 = data_bolsig[:, 1]
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.legend(['BOLOS', 'BOLSIG+'])
plt.xlabel("E/N (Td)")
plt.ylabel("Mean Energy (eV)")
plt.savefig(f"bolos_v_bolsig+.pdf")

# %%
data_bolos_rates = np.loadtxt('bolos_nepc_rates.txt', skiprows=5)
data_MW_rates = np.loadtxt('N2_Rates_MW.txt', skiprows=5)

x1 = data_bolos_rates[:, 0]
y1 = data_bolos_rates[:, 1] * 1e+06
x2 = data_MW_rates[:, 0]
y2 = data_MW_rates[:, 1]
plt.loglog(x1, y1)
plt.loglog(x2, y2)
plt.legend(['BOLOS Ratefile', 'N2_MW Ratefile'])
plt.xlabel("Mean Energy (eV)")
plt.ylabel("Effective (cm^3/s) ")
plt.savefig(f"compare_effective.pdf")
# %%
