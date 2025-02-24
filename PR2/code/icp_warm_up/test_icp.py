import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import cKDTree


if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)
  yaw_angles = np.linspace(0, 2*np.pi, num=36)  # 10-degree steps (adjust as needed)
  for i in range(num_pc):
    print(f"number of point clouds: {i+1}")
    target_pc = load_pc(obj_name, i)


    pose = np.eye(4)

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)

