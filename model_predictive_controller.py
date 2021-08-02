#!/usr/bin/env python3
"""
Model Predictive Control algorithm for linear and angular speed control.

Originally modified from https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py  # noqa: E501
"""

from dataclasses import dataclass
from collections import namedtuple
from datetime import datetime
from typing import Optional, Tuple, Union, cast

import cvxpy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.transforms as tr
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# State vector length
# z = [x, y, v, phi, w]
# x, y position
# v velocity in the direction phi
# w (omega) angular speed
NX = 5
# Input vector dimensions
# u =[a, alpha]
# Acceleration a in the direction phi (dv/dt)
# Angular acceleration alpha
NU = 2

MPC_DEFAULTS = {}

# Horizon length (how many samples forward in time to predict motion)
# T * DT gives the horizon length in seconds
MPC_DEFAULTS["T"] = 5

# MPC parameters
# Using the following cost function where:
#   z is the current state (at time t)
#   z_ref is the current reference state
#   z_T is the predicted state at the horizon
#   z_T,ref is the reference horizon state
#   u is the current inputs
#   du is the change in inputs between times t and t + 1
#   ' denotes transpose, @ matrix multiplication
#   Vectors are assumed column vectors
#   Let quad(v, A) = v' @ A @ v:
#
#       J = quad(z_T,ref - z_T, Qf) + quad(z_ref - z, Q) + quad(u, R) + quad(du, Rd)
MPC_DEFAULTS["R"] = np.diag([0.1, 0.01])  # input cost matrix
MPC_DEFAULTS["Rd"] = np.diag([0.01, 0.01])  # input difference cost matrix
MPC_DEFAULTS["Q"] = np.diag([1.0, 1.0, 0.5, 0.5, 0.5])  # state cost matrix
MPC_DEFAULTS["Qf"] = MPC_DEFAULTS["Q"]  # state final matrix

# Iterative MPC parameters
MPC_DEFAULTS["MAX_ITERATIONS"] = 3
MPC_DEFAULTS["DU_THRESHOLD"] = 0.1  # Condition for halting iteration

MPC_DEFAULTS["DT"] = 0.2  # [s] Time discretisation interval
MPC_DEFAULTS["DL"] = 1.0  # [m] Distance discretisation interval

MPC_DEFAULTS["NEAREST_POINT_SEARCH_DISTANCE"] = 10

# Simulation parameters
MAX_TIME = 100.0
WHEEL_WIDTH = 0.2
MPC_DEFAULTS["TERM_DIST"] = 1.5  # goal / termination distance
MPC_DEFAULTS["STOP_SPEED"] = 0.1  # stop speed


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class RobotProperties(
    namedtuple(
        "RobotProperties",
        [
            "max_forward_velocity",
            "min_forward_velocity",
            "max_angular_velocity",
            "max_forward_acceleration",
            "min_forward_acceleration",
            "max_angular_acceleration",
            "total_width",
            "total_length",
            "wheel_radius",
            "wheel_width",
            "wheel_separation_width",
            "wheel_separation_length",
        ],
    )
):
    __slots__ = ()

    @property
    def theta(self):
        return np.arctan(self.wheel_separation_length / self.wheel_separation_width)

    @property
    def wheel_separation_centre(self):
        return np.linalg.norm(
            [self.wheel_separation_width / 2, self.wheel_separation_length / 2]
        )


ROBOT_PROPERTIES = RobotProperties(
    max_angular_velocity=1.40,
    max_angular_acceleration=0.15,
    max_forward_velocity=1.0,
    min_forward_velocity=-0.25,
    max_forward_acceleration=1.40,
    min_forward_acceleration=-0.5,
    total_width=1.3,
    total_length=1.4,
    wheel_radius=0.15,
    wheel_width=0.2,
    wheel_separation_width=1.0,
    wheel_separation_length=1.1,
)


@dataclass
class RoverState:
    """
    Store and update rover state vector.
    """

    x: float
    y: float
    _v: float
    _phi: float
    _w: float
    drivetrain_params: RobotProperties

    def __post_init__(self):
        self.max_v: float = self.drivetrain_params.max_forward_velocity
        self.min_v: float = self.drivetrain_params.min_forward_velocity
        self.max_w: float = self.drivetrain_params.max_angular_velocity
        self.max_a: float = self.drivetrain_params.max_forward_acceleration
        self.min_a: float = self.drivetrain_params.min_forward_acceleration
        self.max_alpha = self.drivetrain_params.max_angular_acceleration

        self.v = self._v
        self.w = self._w
        self.phi = self._phi

    @property
    def vector(self) -> FloatArray:
        return np.array([self.x, self.y, self.v, self.phi, self.w])

    @vector.setter
    def vector(self, vec: FloatArray):
        self.x = vec[0]
        self.y = vec[1]
        self.v = vec[2]
        self.phi = vec[3]
        self.w = vec[4]

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, v: float):
        self._v = cast(float, np.clip(v, self.min_v, self.max_v))

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float):
        self._w = cast(float, np.clip(w, -self.max_w, self.max_w))

    @property
    def phi(self) -> float:
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = wrap_angle(phi)

    def update_state(self, u: FloatArray, dt: float) -> FloatArray:
        """
        Use non-linear state-space model to update the state.
        """
        a = np.clip(u[0], self.min_a, self.max_a)
        alpha = np.clip(u[1], -self.max_alpha, self.max_alpha)

        self.x += self.v * np.cos(self.phi) * dt
        self.y += self.v * np.sin(self.phi) * dt
        self.v += a * dt
        self.phi += self.w * dt
        self.w += alpha * dt

        return self.vector

    def predict_motion(self, u: FloatArray, time_steps: int, dt: float) -> FloatArray:
        """
        Predict motion for a number of time steps given a set of inputs u for
        each time step from the current state. This does not mutate the state.
        """
        z_bar = np.zeros((NX, time_steps + 1))
        z_bar[:, 0] = self.vector

        state = self.copy()
        for t in range(0, time_steps):
            z_bar[:, t + 1] = state.update_state(u[:, t], dt)

        return z_bar

    @staticmethod
    def get_linear_model(
        v: float, phi: float, dt: float
    ) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """
        Return linearised discrete-time model using first-order Taylor expansion
        around (v, phi).
        """

        A = np.array(
            [
                [1, 0, np.cos(phi) * dt, -v * np.sin(phi) * dt, 0],
                [0, 1, np.sin(phi) * dt, v * np.cos(phi) * dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1],
            ]
        )

        B = np.array([[0, 0], [0, 0], [dt, 0], [0, 0], [0, dt]])

        C = np.array([v * np.sin(phi) * phi * dt, -v * np.cos(phi) * phi * dt, 0, 0, 0])

        return A, B, C

    def copy(self) -> "RoverState":
        return RoverState(*self.vector, drivetrain_params=self.drivetrain_params)


class ModelPredictiveController:
    """
    Model predictive control utilises a kinematic model of the rover to predict its
    motion, then uses a quadratic optimiser to find the optimal inputs within
    constraints on velocity and acceleration.
    """

    def __init__(
        self,
        drivetrain_params: RobotProperties,
        TARGET_SPEED: Optional[float] = None,
        DT: Optional[float] = None,
        DL: Optional[float] = None,
        T: Optional[int] = None,
        R: Optional[FloatArray] = None,
        Rd: Optional[FloatArray] = None,
        Q: Optional[FloatArray] = None,
        Qf: Optional[FloatArray] = None,
        TERM_DIST: Optional[float] = None,
        NEAREST_POINT_SEARCH_DISTANCE: Optional[int] = None,
        MAX_ITERATIONS: Optional[int] = None,
        DU_THRESHOLD: Optional[float] = None,
        STOP_SPEED: Optional[float] = None,
    ):
        self.T: int = T or MPC_DEFAULTS["T"]
        self.R: FloatArray = R or MPC_DEFAULTS["R"]
        self.Rd: FloatArray = Rd or MPC_DEFAULTS["Rd"]
        self.Q: FloatArray = Q or MPC_DEFAULTS["Q"]
        self.Qf: FloatArray = Qf or MPC_DEFAULTS["Qf"]
        self.TERM_DIST: float = TERM_DIST or MPC_DEFAULTS["TERM_DIST"]

        self.drivetrain_params: RobotProperties = drivetrain_params

        self.MAX_SPEED: float = self.drivetrain_params.max_forward_velocity
        self.MIN_SPEED: float = self.drivetrain_params.min_forward_velocity
        self.MAX_OMEGA: float = self.drivetrain_params.max_angular_velocity
        self.MAX_ACCEL: float = self.drivetrain_params.max_forward_acceleration
        self.MIN_ACCEL: float = self.drivetrain_params.min_forward_acceleration
        self.MAX_ALPHA: float = self.drivetrain_params.max_angular_acceleration

        self.TARGET_SPEED: float = TARGET_SPEED or self.MAX_SPEED
        self.STOP_SPEED: float = STOP_SPEED or MPC_DEFAULTS["STOP_SPEED"]
        self.DT: float = DT or MPC_DEFAULTS["DT"]
        self.DL: float = DL or MPC_DEFAULTS["DL"]

        self.MAX_ITERATIONS: int = MAX_ITERATIONS or MPC_DEFAULTS["MAX_ITERATIONS"]
        self.DU_THRESHOLD: float = DU_THRESHOLD or MPC_DEFAULTS["DU_THRESHOLD"]
        self.NEAREST_POINT_SEARCH_DISTANCE: float = (
            NEAREST_POINT_SEARCH_DISTANCE
            or MPC_DEFAULTS["NEAREST_POINT_SEARCH_DISTANCE"]
        )

        self.state: RoverState = RoverState(0, 0, 0, 0, 0, self.drivetrain_params)

        self.z: cvxpy.Variable = cvxpy.Variable((NX, self.T + 1))
        self.u: cvxpy.Variable = cvxpy.Variable((NU, self.T))
        self.z_ref: cvxpy.Parameter = cvxpy.Parameter((NX, self.T + 1))
        self.initial_state: cvxpy.Parameter = cvxpy.Parameter((NX,))

        self._init_optimiser()
        self._init_transform_matrix()

    def prepare(self, start: FloatArray, goal: FloatArray):
        self._generate_straight_trajectory(start, goal)
        self._fill_transform_matrix(*start, self.transform_angle)
        self.target_index: int = 0
        self.last_input: FloatArray = np.zeros((NU, self.T))

    def calculate_control_signal(
        self, unsafe_current_state: FloatArray, full: bool = False
    ) -> Union[Tuple[FloatArray, bool], Tuple[FloatArray, FloatArray, bool]]:
        # Unsafe as it may contain invalid values that will break the optimiser
        self.state.vector = self.apply_transform(unsafe_current_state)
        z_ref, self.target_index = self._calculate_reference_trajectory(
            self.target_index
        )
        output_z, output_u, success = self._iterative_linear_mpc_control(
            z_ref, self.state.vector
        )

        if full:
            return output_u, self.apply_inverse_transform(output_z), success
        else:
            return output_u[:, 0], success

    def _iterative_linear_mpc_control(
        self, z_ref: FloatArray, initial_state: FloatArray
    ) -> Tuple[FloatArray, FloatArray, bool]:
        """
        Perform MPC control, iteratively updating the operating point.
        """
        output_z = cast(FloatArray, None)
        output_u = cast(FloatArray, None)
        success = cast(bool, None)
        for _ in range(self.MAX_ITERATIONS):
            z_bar = self.state.predict_motion(self.last_input, self.T, self.DT)
            output_z, output_u, success = self._linear_mpc_control(
                z_ref, z_bar, initial_state
            )
            if not success:
                break
            du = np.sum(np.abs(output_u - self.last_input))
            self.last_input = output_u
            if du <= self.DU_THRESHOLD:
                break

        return output_z, output_u, success

    def _linear_mpc_control(
        self,
        z_ref: FloatArray,
        z_bar: FloatArray,
        initial_state: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, bool]:
        self.z_ref.value = z_ref
        self.initial_state.value = initial_state
        constraints = self.constraints.copy()

        for t in range(self.T):
            A, B, C = RoverState.get_linear_model(z_bar[2, t], z_bar[3, t], self.DT)

            constraints.append(
                self.z[:, t + 1] == A @ self.z[:, t] + B @ self.u[:, t] + C
            )

        prob = cvxpy.Problem(cvxpy.Minimize(self.cost), constraints)
        solver_success = True
        try:
            prob.solve(solver=cvxpy.OSQP, verbose=False)
        except cvxpy.SolverError:
            print("ERROR: OSQP solving failed, retrying with verbose=True")
            try:
                prob.solve(solver=cvxpy.OSQP, verbose=True)
            except cvxpy.SolverError:
                solver_success = False

        if (
            solver_success
            or prob.status == cvxpy.OPTIMAL
            or prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            output_z = self.z.value
            output_u = self.u.value
            success = True
        else:
            print(
                f"ERROR: MPC optimisation failed with status: {prob.status}. "
                + "This should not happen."
            )
            print("Input dump:")
            print(f"Reference trajectory: {z_ref}")
            print(f"Linearisation point: {z_bar}")
            print(
                f"Initial state (are these within acceptable bounds?): {initial_state}"
            )
            output_z = np.zeros((NX, self.T + 1))
            output_u = np.zeros((NU, self.T))
            success = False

        # Optimisation shouldn't be able to fail, but we need the
        # success result because if it does, we don't want everything
        # to crash and burn
        return output_z, output_u, success

    def _init_transform_matrix(self):
        # This will look like
        # A =
        #   [ cos(-phi0), sin(-phi0), 0, 0, 0,  -x0*cos(-phi0) + y0*sin(-phi0)]
        #   [-sin(-phi0), cos(-phi0), 0, 0, 0,  -x0*sin(-phi0) - y0*cos(-phi0)]
        #   [          0,          0, 1, 0, 0,                               0]
        #   [          0,          0, 0, 1, 0,                           -phi0]
        #   [          0,          0, 0, 0, 1,                               0]
        #   [          0,          0, 0, 0, 0,                               1]
        self.transform_matrix = np.eye(NX + 1)

        # This will look like
        # A^-1 =
        #   [cos(phi0), -sin(phi0), 0, 0, 0,   x0]
        #   [sin(phi0),  cos(phi0), 0, 0, 0,   y0]
        #   [        0,          0, 1, 0, 0,    0]
        #   [        0,          0, 0, 1, 0, phi0]
        #   [        0,          0, 0, 0, 1,    0]
        #   [        0,          0, 0, 0, 0,    1]
        self.inverse_transform_matrix = np.eye(NX + 1)

    def _fill_transform_matrix(self, x0, y0, phi0):
        # Fill top left block with 2D rotation matrix through -phi0
        c = np.cos(-phi0)
        s = np.sin(-phi0)
        self.transform_matrix[0:2, 0:2] = [[c, -s], [s, c]]
        # Fill the last column with the bias corresponding to translation
        # through the rotated origin and rotation by -phi0
        self.transform_matrix[:-2, -1:] = [
            [-x0 * c + y0 * s],
            [-x0 * s - y0 * c],
            [0],
            [-phi0],
        ]

        self.inverse_transform_matrix[0:2, 0:2] = [
            [np.cos(phi0), -np.sin(phi0)],
            [np.sin(phi0), np.cos(phi0)],
        ]

        self.inverse_transform_matrix[:-2, -1:] = [[x0], [y0], [0], [phi0]]

    def augment(self, z):
        shape = z.shape
        if len(shape) == 1:
            rowshape = (1,)
        else:
            rowshape = (1, shape[1])
        return np.append(z, np.ones(rowshape), axis=0)

    def deaugment(self, z):
        if np.all(z[-1] == 1):
            return z[:-1]
        else:
            raise ValueError("input not augmented")

    def apply_transform(self, z):
        t = self.transform_matrix @ self.augment(z)
        t[3] = wrap_angle(t[3])
        return self.deaugment(t)

    def apply_inverse_transform(self, z):
        t = self.inverse_transform_matrix @ self.augment(z)
        t[3] = wrap_angle(t[3])
        return self.deaugment(t)

    def _init_optimiser(self):
        self.state_cost: cvxpy.Expression = cast(cvxpy.Expression, 0.0)
        self.input_cost: cvxpy.Expression = cast(cvxpy.Expression, 0.0)
        self.input_change_cost: cvxpy.Expression = cast(cvxpy.Expression, 0.0)

        self.terminal_cost: cvxpy.Expression = cvxpy.quad_form(
            self.z_ref[:, self.T] - self.z[:, self.T], self.Qf
        )

        for t in range(self.T):
            if t >= 1:
                self.state_cost += cvxpy.quad_form(
                    self.z_ref[:, t] - self.z[:, t], self.Q
                )
                self.input_change_cost += cvxpy.quad_form(
                    self.u[:, t] - self.u[:, t - 1], self.Rd
                )

            self.input_cost += cvxpy.quad_form(self.u[:, t], self.R)

        self.cost: cvxpy.Expression = (
            self.terminal_cost
            + self.state_cost
            + self.input_cost
            + self.input_change_cost
        )

        self.constraints = [
            self.z[:, 0] == self.initial_state,
            self.z[2, :] <= self.MAX_SPEED,
            self.z[2, :] >= self.MIN_SPEED,
            cvxpy.abs(self.z[4, :]) <= self.MAX_OMEGA,
            self.u[0, :] <= self.MAX_ACCEL,
            self.u[0, :] >= self.MIN_ACCEL,
            cvxpy.abs(self.u[1, :]) <= self.MAX_ALPHA,
        ]

    def _calculate_reference_trajectory(
        self, target_index: int
    ) -> Tuple[FloatArray, int]:
        """
        Get the points on the target trajectory up to the horizon.
        """
        z_ref = np.zeros((NX, self.T + 1))

        index = self._get_nearest_reference_point(target_index)

        # todo: find vectorised method for this
        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(self.state.v) * self.DT
            d_index = int(round(travel / self.DL))

            if (index + d_index) < self.num_trajectory_steps:
                z_ref[:, i] = self.trajectory[:, index + d_index]
            else:
                z_ref[:, i] = self.trajectory[:, -1]

        return z_ref, index

    def _generate_straight_trajectory(self, start: FloatArray, end: FloatArray):
        diff = end - start
        self.transform_angle = np.arctan2(diff[1], diff[0])

        length = np.linalg.norm(diff)
        self.num_trajectory_steps = int(np.ceil(length / self.DL))

        # The affine transform aligns the straight trajectory with the x-axis,
        # so y-axis values map to 0.
        # It maps phi to the difference between the reference trajectory and the
        # current one. The goal is for this to be 0.
        # The angular velocity profile is constant (because it is a straight
        # line), so it is 0.
        # If it isn't, we can replace it with
        #    np.diff(angles, append=0) / self.DT

        x_steps = np.linspace(0, length, self.num_trajectory_steps)
        speeds = np.append(
            self.TARGET_SPEED * np.ones(self.num_trajectory_steps - 1), 0
        )

        self.trajectory = np.zeros((NX, self.num_trajectory_steps))

        self.trajectory[0, :] = x_steps
        self.trajectory[2, :] = speeds

    def _get_nearest_reference_point(self, target_index: int) -> int:
        """
        Returns the index and distance from the nearest point on the reference
        trajectory.
        """
        xy_ref = self.trajectory[0:2, :].reshape((2, -1))
        xy = cast(FloatArray, self.state.vector[0:2])
        dists = np.linalg.norm(
            xy.reshape((2, 1))
            - xy_ref[
                :,
                target_index : (  # noqa: E203
                    target_index + self.NEAREST_POINT_SEARCH_DISTANCE
                ),
            ],
            axis=0,
        )
        min_index = np.argmin(dists)
        index = target_index + min_index

        return cast(int, index)


class MPCSimulator:
    def __init__(
        self,
        mpc: ModelPredictiveController,
        robot_props: RobotProperties,
        initial_state: RoverState,
        goal: FloatArray,
        show_animation: bool = True,
        show_time: bool = True,
        save_animation: bool = False,
    ):
        self.mpc: ModelPredictiveController = mpc
        self.props: RobotProperties = robot_props
        self.initial_state: RoverState = initial_state
        self.goal: FloatArray = goal
        self.show_animation: bool = show_animation
        self.show_time: bool = show_time
        self.save_animation: bool = save_animation and show_animation
        self.output_z: Optional[FloatArray] = None

    def _init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.lines = {}
        self.lines["pred"] = self.ax.plot([], [], "xr", label="MPC predicted path")[0]
        self.ref_path = self.mpc.apply_inverse_transform(self.mpc.trajectory)
        self.lines["ref"] = self.ax.plot(
            self.ref_path[0, :],
            self.ref_path[1, :],
            "-r",
            label="Reference path",
        )[0]
        self.lines["traj"] = self.ax.plot([], [], "ob", label="Trajectory")[0]
        self.lines["z_ref"] = self.ax.plot([], [], "xk", label="Reference points")[0]
        self.lines["targ"] = self.ax.plot([], [], "xg", label="Target")[0]
        self._init_car()

        self.ax.grid(True)
        self.ax.set_title("Time[s]:, speed[m/s]:")
        self.ax.legend()
        lx, rx = sorted([self.state.x, self.goal[0]])
        ly, ry = sorted([self.state.y, self.goal[1]])
        self.ax.set_xlim(lx - 5, rx + 5)
        self.ax.set_ylim(ly - 5, ry + 5)
        self.ax.set_aspect("equal")

        self.fig.canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

    def _init_car(self):
        (
            car_anchor,
            car_transform,
            wheel_anchors,
            wheel_transforms,
        ) = self._compute_coordinates_and_transforms()
        # Draw the car in 2D (horizontally orientated, so forward is +x-axis)
        self.car_chassis = patches.Rectangle(
            car_anchor,
            self.props.total_length,
            self.props.total_width,
            transform=car_transform + self.ax.transData,
            fill=False,
        )

        self.car_wheels = []
        for anchor, transform in zip(wheel_anchors, wheel_transforms):
            self.car_wheels.append(
                patches.Rectangle(
                    anchor,
                    2 * self.props.wheel_radius,
                    self.props.wheel_width,
                    transform=transform,
                    fill=False,
                )
            )

        self.ax.add_patch(self.car_chassis)
        for w in self.car_wheels:
            self.ax.add_patch(w)

        self.car_centre = self.ax.plot(self.state.x, self.state.y, "*")[0]

    def _compute_coordinates_and_transforms(self):
        car_anchor = (
            self.state.x - self.props.total_length / 2,
            self.state.y - self.props.total_width / 2,
        )
        # This is not commutative: transform first in data coordinates, then convert
        # to axes coordinates
        car_transform = tr.Affine2D().rotate_around(
            self.state.x, self.state.y, self.state.phi
        )

        wheel_centres = [
            (
                self.state.x + i * self.props.wheel_separation_length / 2,
                self.state.y + j * self.props.wheel_separation_width / 2,
            )
            for i, j in zip([1, 1, -1, -1], [1, -1, -1, 1])
        ]

        wheel_anchors = [
            (
                wheel_centres[i][0] - self.props.wheel_radius,
                wheel_centres[i][1] - self.props.wheel_width / 2,
            )
            for i in range(4)
        ]

        _, angles = self._calculate_inverse_kinematics()

        wheel_transforms = []
        for centre, angle in zip(wheel_centres, angles):
            wheel_transforms.append(
                tr.Affine2D().rotate_around(*centre, angle)
                + car_transform
                + self.ax.transData
            )

        return car_anchor, car_transform, wheel_anchors, wheel_transforms

    def _animate_car(self):
        (
            car_anchor,
            car_transform,
            wheel_anchors,
            wheel_transforms,
        ) = self._compute_coordinates_and_transforms()
        # Draw the car in 2D (horizontally orientated, so forward is +x-axis)
        self.car_chassis.set_xy(car_anchor)
        self.car_chassis.set_transform(car_transform + self.ax.transData)

        for i in range(4):
            self.car_wheels[i].set_xy(wheel_anchors[i])
            self.car_wheels[i].set_transform(wheel_transforms[i])

        self.car_centre.set_data(self.state.x, self.state.y)

    def _calculate_inverse_kinematics(
        self,
    ) -> Tuple[FloatArray, FloatArray]:
        # Normalise
        kv = self.state.v / self.mpc.MAX_SPEED
        kw = self.state.w / (self.mpc.MAX_SPEED / self.props.wheel_separation_centre)

        # half pi, theta translated, theta
        hp, pt, p = np.pi / 2, self.props.theta - np.pi / 2, self.props.theta
        B = np.sin(np.array([[hp, pt, p], [hp, -pt, p], [hp, -pt, -p], [hp, pt, -p]]))

        K = np.array([[kv, kw, 1.0j * kw]]).T
        wheel_vecs = B @ K

        wheel_v = np.abs(wheel_vecs)
        max_v = max(wheel_v)
        if max_v > 1:
            wheel_v /= max_v

        wheel_angles = np.array([np.angle(z) for z in wheel_vecs])

        return wheel_v, wheel_angles

    def _update_title(self):
        self.ax.set_title(f"Time[s]: {self.time:.1f}, speed[m/s]: {self.state.v:0.2f}")

    def _animate(self):
        z_ref_tf, _ = self.mpc._calculate_reference_trajectory(self.mpc.target_index)
        z_ref = self.mpc.apply_inverse_transform(z_ref_tf)
        if self.output_z is not None:
            self.lines["pred"].set_data(self.output_z[0, :], self.output_z[1, :])

        self.lines["traj"].set_data(self.state_history[0, :], self.state_history[1, :])
        self.lines["z_ref"].set_data(z_ref[0, :], z_ref[1, :])
        self.lines["targ"].set_data(
            self.ref_path[0, self.mpc.target_index],
            self.ref_path[1, self.mpc.target_index],
        )
        self._animate_car()
        self._update_title()

        return (
            self.lines["pred"],
            self.lines["traj"],
            self.lines["z_ref"],
            self.lines["targ"],
            self.car_chassis,
            *self.car_wheels,
            self.car_centre,
        )

    def check_goal(self) -> bool:
        d = np.linalg.norm(self.state.vector[0:2] - np.array(self.goal))

        isgoal = d <= self.mpc.TERM_DIST

        isstop = abs(self.state.v) <= self.mpc.STOP_SPEED

        if isgoal and isstop:
            return True

        return False

    def do_simulation(
        self,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        state_vector = cast(FloatArray, self.initial_state.vector)
        self.state_history: FloatArray = state_vector.reshape((NX, 1))
        self.input_history: FloatArray = np.zeros((NU, 1))
        self.time_history: FloatArray = np.zeros((1,))
        self.cost_history: FloatArray = np.zeros((1,))
        self.cost_vars_history: FloatArray = np.zeros((4, 1))
        self.error_history: FloatArray = np.zeros((NX, 1))
        self.state: RoverState = self.initial_state

        self.mpc.prepare(state_vector[0:2], self.goal)

        self.time: float = 0.0
        times = []

        if self.show_animation:
            self._init_plot()
            plt.show(block=False)
            plt.pause(0.1)
            bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            if self.save_animation:
                writer = FFMpegWriter(fps=2 / self.mpc.DT)
                writer.setup(self.fig, "model_predictive_controller.gif")

        while self.time <= MAX_TIME:
            start_time = datetime.now()
            self.output_u, self.output_z, success = cast(
                Tuple[FloatArray, FloatArray, bool],
                self.mpc.calculate_control_signal(self.state.vector, full=True),
            )
            times.append(datetime.now() - start_time)
            if not success:
                print("WARNING: optimisation failed!")

            if self.show_time:
                print(f"\rElapsed: {times[-1]}", end="")

            self.state.update_state(self.output_u[:, 0], self.mpc.DT)
            self.time += self.mpc.DT

            self.state_history = np.append(
                self.state_history,
                cast(FloatArray, self.state.vector).reshape((NX, 1)),
                axis=1,
            )
            self.input_history = np.append(
                self.input_history, self.output_u[:, 0].reshape((NU, 1)), axis=1
            )
            self.time_history = np.append(self.time_history, self.time)
            self.cost_history = np.append(self.cost_history, self.mpc.cost.value)
            self.cost_vars_history = np.append(
                self.cost_vars_history,
                np.array(
                    [
                        [self.mpc.terminal_cost.value],
                        [self.mpc.state_cost.value],
                        [self.mpc.input_cost.value],
                        [self.mpc.input_change_cost.value],
                    ]
                ),
                axis=1,
            )
            self.error_history = np.append(
                self.error_history,
                (
                    self.mpc.z_ref[:, 0].value
                    - self.mpc.apply_transform(self.state.vector)
                ).reshape((NX, 1)),
                axis=1,
            )

            if self.show_animation:
                self.fig.canvas.restore_region(bg)
                for artist in self._animate():
                    self.ax.draw_artist(artist)
                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()

                if self.save_animation:
                    writer.grab_frame()
                # Try to update in as real-time as possible
                plt.pause(
                    max(
                        0.01,
                        self.mpc.DT - (datetime.now() - start_time).total_seconds(),
                    )
                )

            if self.check_goal():
                # Clear the whole line (in case show_time is set to True)
                print("\33[2K\rGoal reached!")
                break

        if self.save_animation:
            writer.finish()

        if self.show_time:
            print(f"Mean: {np.mean(times)}")

        return (
            self.time_history,
            self.state_history,
            self.input_history,
            self.cost_history,
            self.cost_vars_history,
            self.error_history,
        )


def main():
    print("Starting MPC simulation")
    show_plots = True

    mpc = ModelPredictiveController(ROBOT_PROPERTIES)
    initial_state = RoverState(
        x=0.,
        y=0.,
        _v=0.2,
        _phi=1.5,
        _w=0,
        drivetrain_params=mpc.drivetrain_params,
    )
    goal = np.array([10, 5])
    sim = MPCSimulator(mpc, ROBOT_PROPERTIES, initial_state, goal, save_animation=True)
    t, z, u, c, cv, e = sim.do_simulation()

    if show_plots:
        plt.close("all")
        plt.figure(1)
        plt.plot(sim.ref_path[0, :], sim.ref_path[1, :], "-r", label="spline")
        plt.plot(z[0, :], z[1, :], "-g", label="tracking")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.grid()
        plt.axis("equal")
        plt.title("Trajectory")
        plt.legend()

        plt.figure(2)
        ax2 = plt.subplot(3, 2, 1)
        ax2.plot(t, z[3, :])
        ax2.set_title("Heading")
        ax2.grid()
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("rad")

        ax3 = plt.subplot(3, 2, 2)
        ax3.plot(t, z[2, :])
        ax3.set_title("Speed")
        ax3.grid()
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("m / s")

        ax4 = plt.subplot(3, 2, 3)
        ax4.plot(t, z[4, :])
        ax4.set_title("Angular speed")
        ax4.grid()
        ax4.set_xlabel("t")
        ax4.set_ylabel("rad / s")

        ax5 = plt.subplot(3, 2, 4)
        ax5.plot(t, u[0, :])
        ax5.set_title("Acceleration")
        ax5.grid()
        ax5.set_xlabel("t (s)")
        ax5.set_ylabel("m / s^2")

        ax6 = plt.subplot(3, 2, 5)
        ax6.plot(t, u[1, :])
        ax6.set_title("Angular acceleration")
        ax6.grid()
        ax6.set_xlabel("t (s)")
        ax6.set_ylabel("rad / s^2")

        plt.tight_layout()

        plt.figure(3)
        ax7 = plt.gca()

        ax7.plot(t, c, label="Total cost")
        ax7.plot(t, cv[0, :], label="Temporal cost")
        ax7.plot(t, cv[1, :], label="State cost")
        ax7.plot(t, cv[2, :], label="Input cost")
        ax7.plot(t, cv[3, :], label="Input change cost")
        ax7.legend()
        ax7.set_title("Cost")
        ax7.set_xlabel("t (s)")
        ax7.set_ylabel("Value")

        plt.figure(4)
        plt.plot(t, e[0, :], label="x")
        plt.plot(t, e[1, :], label="y")
        plt.plot(t, e[2, :], label="v")
        plt.plot(t, e[3, :], label="phi")
        plt.plot(t, e[4, :], label="w")
        plt.legend()
        plt.title("Errors")
        plt.xlabel("t (s)")
        plt.ylabel("Value")

        plt.show()


if __name__ == "__main__":
    main()
