import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.interpolate



class WeightedLinearRegression:
    """This class implements a weighted linear regression."""

    def __init__(self, centres, widths, weights=None):

        if len(centres) != len(widths) or len(centres) == 0:
            raise Exception("No kernels or different lengths")
        else:
            self.centres = centres
            self.widths = widths
            if weights == None or weights.shape != centres.shape:
                self.weights = np.zeros(centres.shape)
            else:
                self.weights = weights
            self.n_kernels = len(centres)

    def train(self, x, y):
        """Compute the weights of the kernels.
        The first one is the offset, the second one the slope.
        """

        if self.n_kernels == 0:
            raise Exception("No kernels")
        else:
            for i in range(self.n_kernels):
                xi = x - self.centres[i]
                psi = np.exp(-self.widths[i] * xi ** 2)
                num = np.sum(x * psi * y)
                dnom = np.sum(x **2 * psi)
                self.weights[i] = num / dnom

    def to_json(self):
        dico = {}
        dico['centres'] = self.centres.tolist()
        dico['widths'] = self.widths.tolist()
        dico['weights'] = self.weights.tolist()
        return dico

    @staticmethod
    def from_json(dico):
        w = WeightedLinearRegression(np.array(dico['centres']), np.array(dico['widths']), \
                np.array(dico['weights']))
        return w


class CanonicalSystem:
    """Implementation of the DMP canonical system as explained in (Pastor et al, 2009)."""

    def __init__(self, pattern='discrete', alpha=6.9):

        self.alpha = alpha
        self.pattern = pattern
        if self.pattern == 'discrete':
            self.compute_state = self.compute_discrete_state
            self.compute_state_list = self.compute_discrete_state_list
        elif self.pattern == 'rythmic':
            self.compute_state = self.compute_rythmic_state
            self.compute_state_list = self.compute_rythmic_state_list
            self.s = 1
        else:
            raise Exception("Invalid pattern type specified.")

    def compute_rythmic_state(self, tau=1.0):
        """Compute the state of the system if rythmic DMP.

        tau float: time period of the motion
        """

        self.phi = (self.t/tau) % (2 * math.pi)

    def compute_rythmic_state_list(self, t, tau=1.0):
        """Return the states of the system if rythmic DMP.

        t float array: time where to compute states
        tau float: time period of the motion
        """

        return (t/tau) % (2 * math.pi)

    def compute_discrete_state(self, tau=1.0):
        """Compute the state of the system if discrete DMP.

        tau float: duration of the motion
        """

        self.s = math.exp(- self.alpha * self.t / tau)

    def compute_discrete_state_list(self, t, tau=1.0):
        """Return the states of the system if discrete DMP.

        t float array: time where to compute states
        tau float: time duration of the motion
        """

        return np.exp(- self.alpha * t / tau)

    def step(self, dt=None, tau=1.0):
        """Update the state of the system."""

        self.t += dt
        self.compute_state(tau=tau)

    def reset(self):
        """Reset the system to time null."""

        self.t = 0.
        self.compute_state() 

    def to_json(self):
        dico = {}
        dico['pattern'] = self.pattern
        dico['alpha'] = self.alpha  
        return dico

    @staticmethod
    def from_json(dico):
        cs = CanonicalSystem(pattern=dico['pattern'], alpha=dico['alpha'])
        return cs


class TransformationSystem:
    """Implements the transformation system as formulated in (Pastor et al, 2009)."""

    # TO DO think about adding static parameters K, D, alpha

    def __init__(self, joint, pattern='discrete', cs=None, K=150, D=25, alpha=6.9, \
            params=None, start=0, goal=1, n_bfs=10, hparam=0.3):
        """
        joint string: name of the joint considered
        pattern string: type of DMP used (rythmic or discrete)
        cs CanonicalSystem: canonical system linked to this transformation system
        K float: spring constant of the system
        D float: damper parameter of the system
        alpha float: alpha constant of the canonical system if none specified
        params WeightedLinearRegression: basis functions of the forcing term if already known
        start float: starting angle of the motion
        goal float: ending angle of the motion
        n_bfs int: number of basis functions to use when training the system
        hparam float: parameter between 0 and 1 used to determine kernels widths when training
        """

        self.K = K
        self.D = D     
        self.cs = cs
        # Init a default canonical system if none is provided
        if cs == None:
            self.cs = CanonicalSystem(pattern=pattern, alpha=alpha)
        if self.cs.pattern == 'discrete':
            self.compute_forcing_term = self.compute_forcing_term_discrete
        else:
            raise Exception("Selected pattern does not exist.")
        self.parameters = params
        self.n_bfs = n_bfs
        self.hparam = hparam
        self.start = start
        self.goal = goal
        self.joint = joint

    def compute_derivates(self, tau=1.0):
        """Compute the state derivates of the system.

        tau float: duration of the motion
        """

        self.yd = self.z/tau
        self.zd = (self.K*(self.goal - self.y) - self.D * self.z  \
                - self.K * (self.goal - self.start) * self.cs.s + self.K * self.ft)/tau

    def compute_forcing_term_discrete(self):
        """Compute the forcing term for a discrete DMP at current state."""

        if self.parameters == None:
            raise Exception("Not enough parameters")
        fd = []
        fn = []
        C = self.parameters.centres
        H = self.parameters.widths
        w = self.parameters.weights
        for i in range(0,self.parameters.n_kernels):
            fd.append(math.exp(-H[i] * (self.cs.s - C[i]) ** 2))
            fn.append(fd[-1] * w[i])     
        self.ft = self.cs.s * math.fsum(fn) / math.fsum(fd)

    def reset(self, tau=1.0):
        """Reset the system to initial state.

        tau float: time duration of the motion
        """

        self.compute_forcing_term()
        self.y = self.start
        self.z = 0
        self.compute_derivates(tau=tau)

    def update_state(self, dt):
        """Euler integration of the system state."""

        self.y += self.yd * dt
        self.z += self.zd * dt

    def step(self, dt, tau=1.0):
        """Update the state of the system.

        dt float: timestep
        tau float: duration of the motion
        """

        self.update_state(dt)
        self.compute_forcing_term()
        self.compute_derivates(tau=tau)

    def train(self, te, ye):
        """Train the system with the given trajectory using LfD."""

        # Set tau to the duration of the training motion
        tau = te[len(te)-1] - te[0]

        # Space linearally the basis functions centres in time
        tc = np.linspace(0,tau,num=self.n_bfs)
        C = self.cs.compute_state_list(tc, tau=tau)

        # Compute the basis functions widths based on hparam attribut
        th = tc + tau/float(self.n_bfs)
        xh = self.cs.compute_state_list(th, tau=tau)
        xh = xh - C
        H = math.log(1/self.hparam) / xh**2

        # Interpolate the trajectory to eliminate some noise
        t = np.arange(0, tau, tau/len(ye))
        y = np.zeros(t.shape)
        dy = np.zeros(y.shape)
        ddy = np.zeros(y.shape)
        
        y_gen = scipy.interpolate.interp1d((te-te[0]), ye)
        for i in range(len(t)):
            y[i] = y_gen(t[i])

        # Compute the trajectory speed and acceleration
        for k in range(1,y.shape[0]-1):
            dy[k] = (y[k+1] - y[k-1])/(t[k+1] - t[k-1])
        dy[0] = dy[1]
        dy[-1] = dy[-2]

        for k in range(1,y.shape[0]-1):
            ddy[k] = (dy[k+1] - dy[k-1])/(t[k+1] - t[k-1])
        ddy[0] = ddy[1]
        ddy[-1] = ddy[-2]

        # Compute the canonical states corresponding to the time list
        s = self.cs.compute_state_list(t, tau=tau)

        # Setting start and goal to trajectory start and end position
        self.start = y[0]
        self.goal = y[-1]

        # Compute forcing term by inverting the system equation
        f = (tau**2 * ddy + self.D * tau * dy)/self.K + (y.T - y[-1]) + s * (y[-1] - y[0])

        # Train the DMP using weighted linear regression
        slwr = WeightedLinearRegression(C, H)
        slwr.train(s, f)
        self.parameters = slwr

        # Only useful for debug purposes
        #self.ye = copy.deepcopy(y)
        #self.te = copy.deepcopy(t)
        #self.fe = f

    def to_json(self):
        dico = {}
        dico['goal'] = self.goal
        dico['start'] = self.start
        dico['forcing_term'] = self.parameters.to_json()
        dico['K'] = self.K
        dico['D'] = self.D
        dico['pattern'] = self.cs.pattern
        return dico

    @staticmethod
    def from_json(dico, name):
        ts = TransformationSystem(name, pattern=dico['pattern'], K=dico['K'], D=dico['D'], \
                params=WeightedLinearRegression.from_json(dico['forcing_term']), \
                start=dico['start'], goal=dico['goal'])
        return ts


class DynamicSystem:
    """Implements the whole DMP formulation of (Pastor et al, 2009)."""

    def __init__(self, cs=None, ts=None, tau=1.0):
        """
        cs CanonicalSystem: canonical system of the DMP (common to every joints)
        ts TransformationSystem list: transformation system of every joints considered
        tau float: time duration (if discrete) or time period (if rythmic) of the motion
        """

        self.ts = {}
        self.set_cs(cs)
        self.set_ts(ts)
        self.tau = tau      

    def reset(self):
        """Reset the DMP to initial state."""

        self.cs.reset()
        for k in self.ts.keys():
            self.ts[k].reset(tau=self.tau)

    def set_cs(self, cs):

        self.cs = copy.deepcopy(cs)
        for dmp in self.ts:
            dmp.cs = self.cs

    def set_ts(self, ts):
        """Create the dictionary of the transformation system.
        The joints names are the keys.
        """

        self.ts = {}
        if ts==None:
            return
        for dmp in ts:
            self.add_dmp(dmp)

    def set_start(self, start):
        """Set the start parameter of some transformation system of the DMP.

        start float dict: starting position as values and joint names as keys
        """

        if len(start) != len(self.ts):
            raise Exception("The number of dmps is different from the number of inputs")
        for i in start.keys():
            self.ts[i].start = start[i]

    def set_goal(self, goal):
        """Set the goal parameter of some transformation system of the DMP.

        goal float dict: ending position as values and joint names as keys
        """

        if len(goal) != len(self.ts):
            raise Exception("The number of dmps is different from the number of inputs")
        for i in goal.keys():
            self.ts[i].goal = goal[i]

    def add_dmp(self, dmp):

        self.ts[dmp.joint] = copy.deepcopy(dmp)
        self.ts[dmp.joint].cs = self.cs

    def step(self, dt):
        """Update the state of the system."""

        self.cs.step(dt, tau=self.tau)
        for k in self.ts.keys():
            self.ts[k].step(dt, tau=self.tau)

    def integrate(self, dt, tf=1.0):
        """Integrate the system and return its trajectory.

        dt float: system timestep
        tf float: total duration of the motion
        """

        self.reset()
        move = {}
        t_list = [0.]
        for k in self.ts.keys():
            move[self.ts[k].joint] = [self.ts[k].start]

        while self.cs.t < tf:
            self.step(dt)
            t_list.append(self.cs.t)
            for k in self.ts.keys():
                move[self.ts[k].joint].append(self.ts[k].y)

        for i in move.keys():
            move[i] = np.array(move[i])

        return np.array(t_list), move

    def train_dmp(self, dmp, t, y):
        """Train a given transformation system.

        dmp string: joint of the transformation system to train
        t float array: time list of the training trajectory
        y float array: angle list of the training trajectory
        """

        self.ts[dmp].train(t, y)

    def train(self, t, pos):
        """Train the whole DMP.

        t float array: time list of the motion
        pos float array dict: motion angles lists using joints as keys
        """

        for k in pos.keys():
            self.train_dmp(k, t, pos[k])

    def to_json(self):
        dico = {}
        dico['tau'] = self.tau
        dico['canonical_system'] = self.cs.to_json()
        dico['transformation_system'] = {}
        for n in self.ts.keys():
            dico['transformation_system'][n] = self.ts[n].to_json()
        return dico
    
    @staticmethod
    def from_json(dico):
        ts = []
        for k in dico['transformation_system'].keys():
            ts.append(TransformationSystem.from_json(dico['transformation_system'][k], k))
        dmp = DynamicSystem(cs=CanonicalSystem.from_json(dico['canonical_system']), \
                ts=ts, tau=dico['tau'])
        return dmp




















