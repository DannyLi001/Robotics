import pickle
import sys
import time 
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="1"
ifile = "data/imu/imuRaw" + dataset + ".p"
vfile = "data/vicon/viconRot" + dataset + ".p"

ts = tic()
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

ts = imud[0].copy()
tau = np.zeros(len(ts) - 1)
acc = imud[1:4].copy()
w = imud[4:].copy()

acc_sens = 300
gyro_sens = 3.33
acc_bias = 0
gyro_bias = 0

def calibrate(ts, acc, w, vicd):
  acc_factor = 3300 / 1023 / acc_sens
  gyro_factor = 3300 / 1023 * np.pi / 180 / gyro_sens

  acc = (acc - acc_bias) * acc_factor
  w = (w - gyro_bias) * gyro_factor

  tau = ts[1:] - ts[:-1]
  return acc, w, tau

acc, w, tau = calibrate(ts, acc, w, vicd)

q0 = np.array([1, 0, 0, 0])
qt = []
qt.append(q0)

euler = []
euler.append(np.array(t3d.euler.quat2euler(qt[-1])))

def fun_f(qt, tau_wt):
  tmp = mul(qt, exp([0, tau_wt / 2]))
  return np.array([tmp[0], tmp[1][0], tmp[1][1], tmp[1][2]])

def exp(q):
  tmp = (q[1:] / np.linalg.norm(q[1:]) * np.sin(np.linalg.norm(q[1:]))).flatten()
  return np.exp(q[0]) * np.array([np.cos(np.linalg.norm(q[1:])), tmp[0], tmp[1], tmp[2]])

def mul(q, p):
  q = np.array(q)
  return [q[0] * p[0] - q[1:] @ p[1:].T, q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])]

for i in range(len(tau)):
  qt.append(fun_f(qt[-1], tau[i] * w[:, i]))
  euler.append(np.array(t3d.euler.quat2euler(qt[-1])))

qt = np.array(qt)
euler = np.array(euler)

ground_truth_R = vicd['rots'].copy()
ts_truth = vicd['ts'][0]
truth_euler = []
for i in range(ground_truth_R.shape[2]):
  truth_euler.append(np.array(t3d.euler.mat2euler(ground_truth_R[:,:,i])))
truth_euler = np.array(truth_euler)


fig, ax = plt.subplots(3, 1, figsize=(5, 15))
# Plot X-axis values
ax[0].plot(ts - ts[0], euler[:,0], label='MEASURED', color='blue')
ax[0].plot(ts_truth - ts_truth[0], truth_euler[:,0], label='TRUTH', color='red')
ax[0].set_title('X-axis comparison')
ax[0].legend()

# Plot Y-axis values
ax[1].plot(ts - ts[0], euler[:,1], label='MEASURED', color='blue')
ax[1].plot(ts_truth - ts_truth[0], truth_euler[:,1], label='TRUTH', color='red')
ax[1].set_title('Y-axis comparison')
ax[1].legend()

# Plot Z-axis values
ax[2].plot(ts - ts[0], euler[:,2], label='MEASURED', color='blue')
ax[2].plot(ts_truth - ts_truth[0], truth_euler[:,2], label='TRUTH', color='red')
ax[2].set_title('Z-axis comparison')
ax[2].legend()

plt.tight_layout()
plt.show()
