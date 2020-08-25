# Python modules
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import math
from scipy import integrate
from timeit import default_timer as timer
from matplotlib.ticker import AutoMinorLocator
plt.rcParams['axes.linewidth'] = 1.5

############################################## 
##### True values for cosmological parameters
############################################## 
omega_m_true = 0.315
omega_de_true = 0.685
H_0_true = 67.4

ndim =4 #number of free parameters
nwalkers = 30 #number of random walkers
nsteps= 50 # number of steps in each walker 
chain_cut = 20 #must < nsteps

################## 
##### Prior limits  
################## 
om_low=0.00
om_high=1.00
omega_de_low=0.00 
omega_de_high=1.00
w_d0_low=-1.8
w_d0_high= 0.5
H_0_low= 40.00
H_0_high=80.00


##################################### 
##### Load the the Supernova data set
###################################### 
data = np.genfromtxt("SNe_Ia_580.txt", dtype=None,names = ['red','musp', 'musperr'],usecols=(1,2,3),unpack=True)
redshift = data['red']
mu = data['musp'] 
mu_error = data['musperr']

########################### 
##### D_L model  
########################### 
def func(z,omega_m, omega_de, w_d0):
	bottom = np.sqrt(omega_m*(1+z)**3+omega_de*(1+z)**(3+3*w_d0))
	return 1/bottom

###################### 
##### Distance modulus  
###################### 
def model_run(redshift,y_dl,distance_modulus,log2,H_0,omega_m,omega_de,w_d0):
	for j in range(0,len(redshift)):
		int_val = integrate.quad(func,0,redshift[j],args=(omega_m,omega_de,w_d0))
		y_dl[j] = (300000/H_0)*(1+redshift[j])*int_val[0]
		log2[j] = math.log(y_dl[j],10)
		distance_modulus[j] = 25+ 5*log2[j]
	return distance_modulus 


############################ 
##### Calculating likelihood   
############################ 
def lnlike(theta,redshift,mu,mu_error):
	sigma=1.0 	 
	omega_m,omega_de,w_d0,H_0 = theta 	 
	y_dl = np.zeros(len(redshift))
	log2 = np.zeros(len(redshift))
	distance_modulus = np.zeros(len(redshift))
	model = model_run(redshift,y_dl,distance_modulus,log2,H_0,omega_m,omega_de,w_d0)
	exponent = -0.5*(np.sum(((mu-model)**2)/(mu_error**(2))))
	return exponent	


def lnprior(theta):
	omega_m, omega_de, w_d0,H_0 = theta
	if om_low < omega_m < om_high and omega_de_low < omega_de < omega_de_high and w_d0_low < w_d0 < w_d0_high and H_0_low < H_0 < H_0_high:
		return 0.0
	else:
		return -np.inf
 
def lnprob(theta,redshift,mu,mu_error):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta,redshift,mu,mu_error)

def get_starting_pos(nwalkers, ndim):
    omega_m_true = 0.315
    omega_de_true = 0.685
    H_0_true = 67.4
    w_d0_true = -1
    pos = [np.asarray([omega_m_true,omega_de_true, w_d0_true, H_0_true]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    return pos


def run_mcmc(data, nsteps =50, nthreads=1, nwalkers = 30, ndim=4): 
    pos = get_starting_pos(nwalkers, ndim=ndim)    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(redshift,mu,mu_error),threads=nthreads)    
    start = timer()
    sampler.run_mcmc(pos, nsteps, progress=True)
    end = timer()    
    print("Computation time: %f s"%(end-start))    
    return sampler

sampler = run_mcmc(redshift)
chain = sampler.chain

def make_chain_plot(chain, chain_cut):
    nsteps = chain.shape[1]
    ndim = chain.shape[2]

    fig, axes = plt.subplots(ndim,1,sharex=True)
    fig.set_size_inches(10, 7)
    
    param_names = ['$\Omega_{{\\rm m}}$', '$\Omega_{{\\rm DE}}$', '$w_{{\\rm DE}}$', '$H_{{\\rm 0}}$']

    for i, (ax1,param_name) in enumerate(zip(axes,param_names)):
        ax1.tick_params(which = 'both', direction = 'in')
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.plot(chain[:,:,i].T,linestyle='-',color='k',alpha=0.3)
        ax1.set_ylabel(param_name)
        ax1.set_xlabel("Step number")
        ax1.set_xlim(0,nsteps)
        ax1.axvline(chain_cut,c='r',linestyle='--')

make_chain_plot(chain, chain_cut)
flat_samples = sampler.get_chain(discard=chain_cut, flat=True)
plt.savefig("New1.png")


###############
##### Plotting  
##############
def make_corner_plot(flat_samples, savefile='corner.png'):
    param_names = ['$\Omega_{{\\rm m}}$', '$\Omega_{{\\rm DE}}$', '$w_{{\\rm DE}}$', '$H_{{\\rm 0}}$']
    
    fig = corner.corner(flat_samples, labels=param_names, quantiles=[0.16, 0.5, 0.84], color = "b",truth_color='r', show_titles=True,title_fmt = '.3f')
    plt.savefig(savefile)

make_corner_plot(flat_samples)
plt.show()


###############################################
##### print the best fit parameters with errors
################################################

def get_best_params(chain):
    ndim = chain.shape[1]
    
    vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain, [16, 50, 84],axis=0)))
    
    param_names = ['omega_m','omega_de','w_d0', 'H_0']    
    param_dict = dict(zip(param_names,vals))    
    return param_dict    
    
best_params = get_best_params(flat_samples)
print(best_params)












