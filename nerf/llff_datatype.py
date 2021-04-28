import numpy as np
import json

def load_pose_json(filename,near,far):
    info = None
    with open(filename, encoding='utf-8') as file:
        info = json.loads(file.read())
    poses = []
    for i in range(len(info["view_centers"])):
        pose = np.zeros((3, 5), dtype=np.float32)
        Rt = np.eye(4, dtype=np.float32)
        for j in range(9):
            Rt[j // 3, j % 3] = info["view_rots"][i][j]
        for j in range(3):
            Rt[j, 3] = info["view_centers"][i][j]
        # Rt = np.linalg.inv(Rt)
        Rt = Rt[:3,:4]
        Rt = np.concatenate([-Rt[:, 1:2], Rt[:, 0:1], -Rt[:, 2:3], Rt[:,3:4]],1)

        pose[0,:4] = Rt[0]
        pose[1,:4] = Rt[2]
        pose[2,:4] = Rt[1]
        
        pose[0, 4] = info["view_res"]["x"]
        pose[1, 4] = info["view_res"]["y"]
        pose[2, 4] = info["cam_params"]["fx"]
        pose = pose.flatten()
        pose = np.concatenate((pose, [near,far])) 
        ## better to know the near and far plane from Unity-DengNianchen. 
        ## Hard to understand the exact meaning (the depth bound of scene geometry as I can see)
        # print(pose.shape)
        poses.append(pose) 
    poses = np.stack(poses)
    
    print(poses.shape)
    return poses

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_json', type=str,
                    help='input training set info')
parser.add_argument('near', type=float,
                    help='scene depth range near')
parser.add_argument('far', type=float,
                    help='scene depth range far')
args = parser.parse_args()

if __name__ == "__main__":
    jsonfile = args.train_json
    near = args.near
    far = args.far
    poses = load_pose_json(jsonfile,near,far)
    np.save(args.train_json + "poses_bounds.npy", poses)
    np.savetxt(args.train_json + "poses_bounds.txt", poses)