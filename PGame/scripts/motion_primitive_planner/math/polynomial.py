"""
This script is adapted from the work of Atsushi Sakai and his PythonRobotics project (https://github.com/AtsushiSakai/PythonRobotics).
"""

import numpy as np

# 4-th order polynomial
class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        if time <= 0:
            raise ValueError("The time duration must be greater than zero to avoid singular matrix issues.")

        # Initialize coefficients from initial conditions
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        # Set up matrix A and vector b to solve for a3 and a4
        A = np.array([
            [3 * time ** 2, 4 * time ** 3],
            [6 * time, 12 * time ** 2]
        ])
        b = np.array([
            vxe - self.a1 - 2 * self.a2 * time,
            axe - 2 * self.a2
        ])

        # Solve the matrix equation for a3 and a4
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            raise ValueError("An error occurred solving for the coefficients: " + str(e))

        self.a3, self.a4 = x

    def calc_point(self, t):
        """ Calculate the position at time t """
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4

    def calc_first_derivative(self, t):
        """ Calculate the velocity at time t """
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    def calc_second_derivative(self, t):
        """ Calculate the acceleration at time t """
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

    def calc_third_derivative(self, t):
        """ Calculate the jerk (rate of change of acceleration) at time t """
        return 6 * self.a3 + 24 * self.a4 * t


# 5-th order polynomial
class QuinticPolynomial:
    
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        if time <= 0:
            raise ValueError("The time duration must be greater than zero.")
        
        # Initialize coefficients based on initial conditions
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        # Setup the matrix A and vector b to solve for a3, a4, a5
        A = np.array([
            [time ** 3,      time ** 4,       time ** 5],
            [3 * time ** 2,  4 * time ** 3,   5 * time ** 4],
            [6 * time,       12 * time ** 2,  20 * time ** 3]
        ])
        b = np.array([
            xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
            vxe - self.a1 - 2 * self.a2 * time,
            axe - 2 * self.a2
        ])
        
        # Solve the linear equation system
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            raise ValueError("An error occurred solving the polynomial coefficients: " + str(e))
        
        self.a3, self.a4, self.a5 = x

    def calc_point(self, t):
        """ Calculate position at time t """
        return (self.a0 + self.a1 * t + self.a2 * t ** 2 +
                self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5)

    def calc_first_derivative(self, t):
        """ Calculate velocity at time t """
        return (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 +
                4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4)

    def calc_second_derivative(self, t):
        """ Calculate acceleration at time t """
        return (2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3)

    def calc_third_derivative(self, t):
        """ Calculate jerk at time t """
        return (6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2)