#Exercise Set 2 


# Problem 1: Random Numbers

import numpy as np
np.random.seed(1986)
state = np.random.get_state() #what does get.state() and set.state() exactly do?
for i in range(3):
    np.random.set_state(state)
    for j in range(2):
        x = np.random.uniform()
        print(f'({i},{j}): x = {x:.3f}')


# Problem 2: Expected Value 

# set parameters
sigma = 3.14
omega = 2
N = 10000
np.random.seed(1986)

#draw random numbers
x = np.random() # why add .normal and stuff in parenthesis?

#define function
def g(x, omega):
    y = x.copy() # what does function .copy do?
    y[x < -omega] = -omega
    y[x > omega] = omega
    return y

#compute mean and variance

mean = np.mean(g(x, omega))
var = np.var(g(x-mean, omega))
print(f'mean = {mean:.5f}, var = {var:.5f}')  # why f' and .5f ?


# Problem 3: Histogram

# a. import  
#%matplotlib inline   # importing not working!!
import matplotlib.pyplot as plt
import numpy as np  
# there is more imported in solutions --> is this really necessary?

# b. plotting figure
def fitting_normal(X,mu_guess,sigma_guess):
    
    # i. normal distribution from guess
    F = norm(loc=mu_guess,scale=sigma_guess)
    
    # ii. x-values
    x_low = F.ppf(0.001)  #numbers in () random? whats ppf?
    x_high = F.ppf(0.999)
    x = np.linspace(x_low,x_high,100)

    # iii. figure
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,F.pdf(x),lw=2)
    ax.hist(X,bins=100,density=True,histtype='stepfilled');
    ax.set_ylim([0,0.5])
    ax.set_xlim([-6,6])

# c. parameters
mu_true = 2
sigma_true = 1
mu_guess = 1
sigma_guess = 2

# d. random draws
X = np.random.normal(loc=mu_true,scale=sigma_true,size=10**6)

# e. figure
try:
    fitting_normal(X,mu_guess,sigma_guess)
except:
    print('failed')


# Problem 4: Import MyFunk
import mymodule
mymodule.myfun(2)

# Problem 5: Exchange Economy

# a. parameters
N = 10000
mu = 0.5
sigma = 0.2
mu_low = 0.1
mu_high = 0.9
beta1 = 1.3
beta2 = 2.1
seed = 1986

# b. draws of random numbers
np.random.seed(seed)
alphas = np.random.uniform(low=mu_low,high=mu_high,size=N)


# c. demand function
# d. excess demand function
# e. find equilibrium function
# f. call find equilibrium function









