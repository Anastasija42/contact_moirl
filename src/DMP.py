"""Dynamic movement primitive used to parametrise a smooth reference path."""
import crocoddyl
import pinocchio as pin
from IPython.display import HTML
import mim_solvers
import numpy as np
import random
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from IRL_utils import *
import time
from Optimization_utils import *

class DMP():
    def __init__(self, args):
        self.basis_num = args['basis_num']
        self.alpha_x = args['alpha_x']
        self.alpha_z = args['alpha_z']
        self.beta_z = args['beta_z']
        self.dt = args['dt']
        self.T = args['T']
        self.model = args['model']
        self.noise_f = args['noise_f']
        self.t = np.linspace(0, self.T*self.dt, self.T+1)
        self.x = np.exp(-self.alpha_x * self.t)
        self.tau = args['tau']
        self.theta = None
        self.psi = self.get_psi()
        A = (np.diag([1]*(self.basis_num+1), -1) + np.diag([-2]*(self.basis_num+2), 0) + np.diag([1]*(self.basis_num+1), 1))[:,1:-1]; R = A.T@A; 
        self.M = np.linalg.inv(R)

    def gaussian_basis(self, x, c, h):
        return np.exp(-h * (x - c)**2)

    def get_psi(self):
        centers = np.linspace(0, 1, self.basis_num)
        widths = np.ones(self.basis_num)* self.basis_num**1.5 / centers[-1]**2
        psi = np.array([self.gaussian_basis(self.x, c, h) for c, h in zip(centers, widths)])
        psi_sum = np.sum(psi, axis=0)
        psi_normalized = psi / psi_sum
        return psi_normalized
    
    def learn_traj(self, xs):
        nq = self.model.nq
        a_z = self.alpha_z
        b_z = self.beta_z
        tau = self.tau
        q_des = xs[:, :nq]
        q_dot_des = xs[:, nq:]
        f_target = tau**2 * np.gradient(q_dot_des, self.t, axis=0) - a_z * (b_z * (q_des[-1] - q_des) - tau * q_dot_des)
        weights, _, _, _ = np.linalg.lstsq((self.psi*self.x).T, f_target, rcond=None)
        f_recon = (weights.T@(self.psi*self.x)).T
        return weights, f_recon
    
    def generate_noisy_traj(self, xs, N):
        xs_set = []
        us_set = []
        q_des = xs[:, :self.model.nq]
        q_dot_des = xs[:, self.model.nq:]
        if self.theta is None:
            self.theta, _ = self.learn_traj(xs)
        for i in range(N):
            f_ =  ((self.theta + self.noise_f*np.random.multivariate_normal(np.zeros(self.basis_num), self.M, self.model.nq).T).T@(self.psi*self.x)).T
            y = [q_des[0]]
            y_dot = [q_dot_des[0]]
            y_ddot = [np.zeros_like(q_dot_des[0])]
            for i in range(1, len(self.t)):
                f_i = f_[i]
                y_ddot.append((self.alpha_z * (self.beta_z * (q_des[-1] - y[-1]) - self.tau * y_dot[-1]) + f_i ) / self.tau**2)
                y_dot.append(y_dot[-1] + y_ddot[-1] * (self.t[i] - self.t[i-1]))
                y.append(y[-1] + y_dot[-1] * (self.t[i] - self.t[i-1]))
            y = np.stack(y); y_dot = np.stack(y_dot); y_ddot = np.stack(y_ddot)
            xs_set.append(np.hstack([y, y_dot]))
            us_set.append(self.model.get_control(y, y_dot, y_ddot))
        return xs_set, us_set

    

    