import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import matplotlib.pyplot as plt
 
def Plot2D(cloud2d):
    figure2D = plt.figure(figsize=(7,7))
    axes2D = plt.axes()
    axes2D.scatter(cloud2d[:, 0], cloud2d[:, 1], alpha=0.5)
    plt.show()

def voxel2Numpy(voxel):
    voxels = voxel.get_voxels()
    # notice that the coordinates are in the voxel grid !
    indices = np.stack(list(vx.grid_index for vx in voxels))
    return indices

def PlotVoxel(input_cloud):
    # input cloud must be np.array here
    input_cloud = (input_cloud - input_cloud.min())/(input_cloud.max() - input_cloud.min() + 10e-6)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(input_cloud)
    o3d.visualization.draw_plotly([cloud])

    cloud.scale(1 / np.max(cloud.get_max_bound() - cloud.get_min_bound()), center=cloud.get_center())

    # try downsampling the point cloud that you have with 3D voxels
    # all the points belonging to the cloud are bucketed into voxels and then each occupied voxel generates exactly one point by averaging all points inside

    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=0.025)
    vec3d = o3d.utility.Vector3dVector(input_cloud)
    # use this to check if your points are inside the voxel
    print(np.asarray(voxel.check_if_included(vec3d)))

    print((voxel.get_voxels()[0]).grid_index)
    fig = go.Figure()
    voxel = np.asarray(voxel.get_voxels())
    points = np.asarray([v.grid_index for v in voxel])
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color='green', opacity=0.5)
    ))
    fig.show()
    
def Plot2DWithBuckets(input_plane, resolution, max_pool_points):
    fig, axs = plt.subplots(1,3)

    for i in range(len(axs)):
        plane = list(input_plane.keys())[i]
        axs[i].scatter(input_plane[plane][0, :, 0], input_plane[plane][0, :, 1], alpha=0.5, color='royalblue', marker='o', label=("Points "+plane.upper()))
        for j in range(resolution + 1): # bucket grid
            axs[i].axvline(x=j/resolution, color='gray', linestyle='--', linewidth=0.5)
            axs[i].axhline(y=j/resolution, color='gray', linestyle='--', linewidth=0.5)

        max_pool_points_plane = max_pool_points[plane][0][max_pool_points[plane][0].sum(dim=-1) != 0]
        axs[i].scatter(max_pool_points_plane[:, 0], max_pool_points_plane[:, 1], color='darkorange', marker='o', label=("Max Points "+plane.upper()))

        axs[i].set_xlim(0,1)
        axs[i].set_ylim(0,1)
        axs[i].legend()

    fig.set_figheight(7)
    fig.set_figwidth(21)
    plt.show()