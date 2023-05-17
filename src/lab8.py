import numpy as np
import os
import pandas as pd
import re
from typing import List

import tensorflow as tf
import typing
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, BatchNormalization, Concatenate, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import mean_absolute_error
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.keras import TqdmCallback


def get_deltaS_deltaTheta_from_xy(x: np.ndarray, y: np.ndarray,
                                  return_delta_theta=True, return_trig=False,
                                  back_fill_theta=False, back_fill_delta_d=False) -> np.ndarray:
    """Converts a sequence of x, y positions to relative distance between sequence steps.
    Can also return the relative angle between sequence steps if desired
    Can also return the sin and cos of the relative angle if desired.

    By default the return sequence is the same length as the input sequence but the first timestep is padded to be zero
    If desired the relative distance and/or angle can be backfilled (index 1 is copied to index 0)

    :param x: ndarray of the x positions. The actual data should be in the last dimension.
    :param y: ndarray of the y positions. The actual data should be in the last dimension.
    :param return_delta_theta: if true will return the relative angle
    :param return_trig: if true will return the sin and cos if the relative angle
    :param back_fill_theta: if true will make the first relative angle be equal to the second
    :param back_fill_delta_d: if true will make the first relative distance equal to the second
    :return: numpy ndarray with relative angle and distance between time steps.
    This will be same length as the input x,y.
    The actual data is stored in the last dimension in the order:
    [delta_d, delta_theta, sin(delta_theta), cos(delta_theta), theta)]
    The first values either set to zero or back filled if back fill is set.
    :rtype: np.ndarray
    """
    reduce_to_2d = False
    if x.ndim != y.ndim:
        raise ValueError("x and y should have same dimensions not {0} and {1} respectively".format(x.ndim, y.ndim))
    if x.ndim < 2:
        x = x[None, ...]
        y = y[None, ...]
        reduce_to_2d = True

    delta_x = np.diff(x)
    delta_y = np.diff(y)

    # get absolute angle between each diff
    theta = np.concatenate([np.zeros(shape=(delta_y.shape[0], 1)), np.arctan2(delta_y, delta_x)], axis=-1)

    # back fill if necessary
    if back_fill_theta:
        theta[..., 0] = theta[..., 1]

    # get the relative distance and angle
    delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2)
    delta_theta = np.diff(theta)

    # get the trig values of the relative angles
    sin_delta_theta = np.sin(delta_theta)
    cos_delta_theta = np.cos(delta_theta)

    deltas = np.concatenate(
        [delta_s[..., None], delta_theta[..., None], sin_delta_theta[..., None], cos_delta_theta[..., None]], axis=-1)

    # pad the beginning with zeros to match original size
    zeros_shape = list(deltas.shape)
    zeros_shape[-2] = 1
    deltas = np.concatenate([np.zeros(zeros_shape), deltas], axis=-2)
    # the first cosine term should be 1 not zero
    deltas[..., 0, 3] = 1

    if back_fill_delta_d:
        deltas[..., 0, 0] = deltas[..., 1, 0]

    # only return what they asked for
    ret_columns = [0]
    if return_delta_theta:
        ret_columns += [1]
    if return_trig:
        ret_columns += [2, 3]
    ret = deltas[..., ret_columns]

    if reduce_to_2d:
        ret = ret[0]
    return ret


def getXYfromDeltas(delta_d: np.ndarray,
                    delta_theta: np.ndarray,
                    initial_conditions: np.ndarray = None) -> np.ndarray:
    """Converts a sequence of relative distances and angle changes to 2D x,y positions
    Assumes the starting location is 0,0 unless initial_conditions are given

    :param delta_d: relative distances between each step in the sequence
    :param delta_theta: relative angles betweeen each step in the sequence
    :param initial_conditions: the starting x,y point(s)
    :return: a numpy ndarray with the x,y positions in the last dimension
    :rtype: np.ndarray
    """
    reduce_to_2d = False
    if delta_d.ndim != delta_theta.ndim:
        raise ValueError(
            "delta_d and delta_theta should have same dimensions not {0} and {1} respectively".format(delta_d.ndim,
                                                                                                      delta_theta.ndim))
    if delta_theta.ndim < 2:
        delta_theta = delta_theta[None, ...]
        delta_d = delta_d[None, ...]
        reduce_to_2d = True
    if initial_conditions is None:
        initial_conditions = np.zeros(shape=(delta_d.shape[0], 3))

    theta = np.concatenate([np.zeros(shape=(delta_theta.shape[0], 1)), np.cumsum(delta_theta, axis=-1)], axis=-1)
    theta += initial_conditions[..., 2:3]

    delta_x = delta_d * np.cos(theta[..., :-1] + delta_theta)
    delta_y = delta_d * np.sin(theta[..., :-1] + delta_theta)

    x = np.cumsum(delta_x, axis=-1) + initial_conditions[..., 0:1]
    y = np.cumsum(delta_y, axis=-1) + initial_conditions[..., 1:2]

    position = np.concatenate([x[..., None], y[..., None]], axis=-1)

    if reduce_to_2d:
        position = position[0]
    return position


def get_samples_from_lists(imuList: List[pd.DataFrame], viList: List[pd.DataFrame],
                           num_samples: int,
                           seq_len: int,
                           input_columns_imu: typing.List[str],
                           input_columns_mag: typing.List[str],
                           output_columns: typing.List[str]):
    x_data_imu: np.ndarray
    x_data_mag: np.ndarray
    y_data: np.ndarray
    return x_data_imu, x_data_mag, y_data


def probability_activation_fn(x: tf.Tensor) -> tf.Tensor:
    """ Apply an activation function assuming the input is parameters for a gaussian. Mostly I need to make sure the
    standard deviation is always positive

    :param x: the parameters of a gaussian distribution, I will assume the first is the mean and the second is the
    standard deviation
    :return: the activated tensor
    """

    output_tensor: tf.Tensor
    return output_tensor


def probability_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Get the negative log likelihood of the predicted gaussian distribution evaluated at the targets

    :param y_true: the true values
    :param y_pred: the prediction but stored as the mean and standard deviation.
    thus this will be twice as big as the truth
    :return: the negative log likelihood of the gaussian probability of the given prediction
    """

    neg_log_likelihood: tf.Tensor
    return neg_log_likelihood


def mae_prob_metric_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Get the mean absolute error of the distance while ignoring the standard deviation

    :param y_true: the true values
    :param y_pred: the prediction but stored as the mean and standard deviation.
    :return: the mean absolute error of the distance
    """
    mae_tensor: tf.Tensor
    return mae_tensor


def build_unscaled_mae_prob_metric_fn(output_bias, output_scaling):
    """
    Returns a function that calculates the unscaled mean absolute error

    :param output_bias: the bias for the output
    :param output_scaling: the scaling for the output
    :return: the callable function
    """
    def mae_unscaled_prob_metric_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Get the unscaled mean absolute error of the distance while ignoring the standard deviation

        :param y_true: the true values
        :param y_pred: the prediction but stored as the mean and standard deviation.
        :return: the mean absolute error of the distance
        """
        mae_tensor: tf.Tensor
        return mae_tensor

    return mae_unscaled_prob_metric_fn


def build_model_fancy(seq_len: int,
                      input_data_size_imu: int,
                      input_data_size_mag: int,
                      output_data_size: int,
                      batch_size: int = None,
                      stateful=False) -> Model:
    """this function build a model

    :param batch_size: the batch size for the model, if not given will default to None
    :param seq_len: the size of the time step dimension
    :param input_data_size_imu: the data size of the input tensor for imu
    :param input_data_size_mag: the data size of the input tensor for mag
    :param output_data_size: the data size of the output tensor
    :param stateful: if true the model will be build stateful
    :return: a build and compiled model
    :rtype: Model
    """

    model: Model
    return model


def main():
    file_root = os.path.join("/opt", "data", "Oxford Inertial Tracking Dataset")

    vicon_column_names: typing.List[str] = "Time Header translation.x translation.y translation.z " \
                                           "rotation.x rotation.y rotation.z rotation.w".split(
        ' ')
    print(vicon_column_names)
    imu_column_names: typing.List[str] = "Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) " \
                                         "rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) " \
                                         "gravity_x(G) gravity_y(G) gravity_z(G) " \
                                         "user_acc_x(G) user_acc_y(G) user_acc_z(G) " \
                                         "magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)".split(
        ' ')
    print(imu_column_names)

    input_columns_imu: typing.List[str] = []
    input_columns_mag: typing.List[str] = []
    output_columns: typing.List[str] = []
    model_name: str = "model.h5"
    num_samples: int
    seq_len: int
    force_retrain_model = False

    epochs: int
    batch_size: int

    # read in raw data
    ignore_first: int = 2000
    vi_list_train: typing.List[pd.DataFrame] = []
    imu_list_train: typing.List[pd.DataFrame] = []
    vi_list_test: typing.List[pd.DataFrame] = []
    imu_list_test: typing.List[pd.DataFrame] = []
    for root, dirs, files in os.walk(file_root, topdown=False):
        if 'handheld' in root and 'syn' in root:
            for i in range(len(files) // 2):
                vi_name = os.path.join(root, f"vi{i + 1}.csv")
                imu_name = os.path.join(root, f"imu{i + 1}.csv")
                if os.path.exists(vi_name) and os.path.exists(imu_name):
                    vi_temp = pd.read_csv(vi_name, names=vicon_column_names)[ignore_first:]
                    imu_temp = pd.read_csv(imu_name, names=imu_column_names)[ignore_first:]

                    deltas = get_deltaS_deltaTheta_from_xy(vi_temp['translation.x'].values,
                                                           vi_temp['translation.y'].values)
                    vi_temp['delta_s'] = deltas[..., 0]
                    vi_temp['delta_theta'] = deltas[..., 1]

                    data_num = int(re.search(r"data(?P<num>\d+)", root).group("num"))
                    if data_num < 5:
                        vi_list_train.append(vi_temp)
                        imu_list_train.append(imu_temp)
                    else:
                        vi_list_test.append(vi_temp)
                        imu_list_test.append(imu_temp)

    print(f"Got {len(vi_list_train)} data frames")

    if not os.path.exists(model_name) or force_retrain_model:
        x_train_imu, x_train_mag, y_train = get_samples_from_lists(imuList=imu_list_train,
                                                                   viList=vi_list_train,
                                                                   num_samples=num_samples,
                                                                   seq_len=seq_len,
                                                                   input_columns_imu=input_columns_imu,
                                                                   input_columns_mag=input_columns_mag,
                                                                   output_columns=output_columns)

        model = build_model_fancy(seq_len=seq_len,
                                  input_data_size_imu=len(input_columns_imu),
                                  input_data_size_mag=len(input_columns_mag),
                                  output_data_size=len(output_columns),
                                  stateful=False)

        # train the model then save

        model.save(model_name)

    new_seq_length: int
    model_one_step = build_model_fancy(seq_len=new_seq_length,
                                       input_data_size_imu=len(input_columns_imu),
                                       input_data_size_mag=len(input_columns_mag),
                                       output_data_size=len(output_columns),
                                       batch_size=1,
                                       stateful=True)

    # note that if you try the function load_model now since we used custom functions it will fail unless
    # you give it references to the custom functions we made via the `custom_objects` dict
    # I prefer this method because it is less prone to failure
    model_one_step.load_weights(model_name)

    x_test_imu: np.ndarray
    x_test_mag: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray

    # Visualization

    # print the whole values
    plt.figure()
    # plot our true delta_s values

    # plot our predicted delta_s values

    # label axes
    plt.title('delta_s Whole Values')
    plt.xlabel('timestep')
    plt.ylabel('distance (m)')
    plt.legend()

    # print the top down view
    plt.figure()
    # plot the true position

    # plot our predicted position

    # label axes
    plt.title('Top Down View')
    plt.xlabel('vicon x axis (m)')
    plt.ylabel('vicon y axis (m)')
    plt.legend()

    # print the error with the standard deviation
    plt.figure()
    # plot the error

    # plot the positive bounds for the standard deviation

    # plot the negative bounds for the standard deviation

    # label axes
    plt.title('delta_s error')
    plt.xlabel('timestep')
    plt.ylabel('distance error (m)')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
