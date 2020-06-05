// System
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>

// Custom
#include <gpg/candidates_generator.h>
#include <gpg/hand_search.h>
#include <gpg/finger_hand.h>
#include <gpg/config_file.h>
#include <gpg/plot.h>

const double PI = M_PI;
const double PI_2 = M_PI*2;

// function to read in a double array from a single line of a configuration file
std::vector<double> stringToDouble(const std::string& str)
{
  std::vector<double> values;
  std::stringstream ss(str);
  double v;

  while (ss >> v)
  {
    values.push_back(v);
    if (ss.peek() == ' ')
    {
      ss.ignore();
    }
  }

  return values;
}

// function to load registrated object pose between mesh and point cloud
Eigen::Affine3d loadRegistration(const std::string& filepath) 
{
  std::ifstream ifs(filepath, std::ifstream::in);
  std::string s;
  ifs >> s;
  Eigen::Matrix4d mesh2cloud;
  mesh2cloud.row(3) << 0,0,0,1;
  std::stringstream ss;
  ss << s;
  std::string substr;
  for (int i=0; i<12; ++i) {
      std::getline(ss, substr, ',');
      mesh2cloud.col(i/3).row(i%3) << std::stod(substr);
  }
  return Eigen::Affine3d(mesh2cloud);
}

int main(int argc, char* argv[])
{
  // Read parameters from configuration file.
  if (argc<5) exit(1);
  ConfigFile config_file(argv[1]);

  double finger_width = config_file.getValueOfKey<double>("finger_width", 0.01);
  double hand_outer_diameter  = config_file.getValueOfKey<double>("hand_outer_diameter", 0.12);
  double hand_depth = config_file.getValueOfKey<double>("hand_depth", 0.06);
  double hand_height  = config_file.getValueOfKey<double>("hand_height", 0.02);
  double init_bite  = config_file.getValueOfKey<double>("init_bite", 0.01);
  double friction_coeff = config_file.getValueOfKey<double>("friction_coeff", 20.0);
  int viable_thresh = config_file.getValueOfKey<int>("viable_thresh", 6);
  double noise_std_cm = config_file.getValueOfKey<double>("noise_std_cm", -1.0);

  std::cout << "finger_width: " << finger_width << "\n";
  std::cout << "hand_outer_diameter: " << hand_outer_diameter << "\n";
  std::cout << "hand_depth: " << hand_depth << "\n";
  std::cout << "hand_height: " << hand_height << "\n";
  std::cout << "init_bite: " << init_bite << "\n";
  std::cout << "friction_coeff: " << friction_coeff << "\n";
  std::cout << "viable_thresh: " << viable_thresh << "\n";
  std::cout << "noise_std_cm: " << noise_std_cm << "\n";

  bool voxelize = config_file.getValueOfKey<bool>("voxelize", true);
  bool remove_outliers = config_file.getValueOfKey<bool>("remove_outliers", false);
  std::string workspace_str = config_file.getValueOfKeyAsString("workspace", "");
  std::string camera_pose_str = config_file.getValueOfKeyAsString("camera_pose", "");
  std::vector<double> workspace = stringToDouble(workspace_str);
  std::vector<double> camera_pose = stringToDouble(camera_pose_str);
  std::cout << "voxelize: " << voxelize << "\n";
  std::cout << "remove_outliers: " << remove_outliers << "\n";
  std::cout << "workspace: " << workspace_str << "\n";
  std::cout << "camera_pose: " << camera_pose_str << "\n";

  int num_samples = config_file.getValueOfKey<int>("num_samples", 1000);
  int num_threads = config_file.getValueOfKey<int>("num_threads", 1);
  double nn_radius = config_file.getValueOfKey<double>("nn_radius", 0.01);
  int num_orientations = config_file.getValueOfKey<int>("num_orientations", 8);
  int rotation_axis = config_file.getValueOfKey<int>("rotation_axis", 2);
  std::cout << "num_samples: " << num_samples << "\n";
  std::cout << "num_threads: " << num_threads << "\n";
  std::cout << "nn_radius: " << nn_radius << "\n";
  std::cout << "num_orientations: " << num_orientations << "\n";
  std::cout << "rotation_axis: " << rotation_axis << "\n";

  bool plot_candidates = config_file.getValueOfKey<bool>("plot_candidates", true);
  bool plot_normals = config_file.getValueOfKey<bool>("plot_normals", false);
  bool output_merged_pcd = config_file.getValueOfKey<bool>("output_merged_pcd", false);
  int max_samples = config_file.getValueOfKey<int>("max_samples", 10);
  std::cout << "plot_candidates: " << plot_candidates << "\n";
  std::cout << "plot_normals: " << plot_normals << "\n";
  std::cout << "max_samples: " << max_samples << "\n";
  std::cout << "output_merged_pcd: " << output_merged_pcd << "\n";

  if (output_merged_pcd && argc<6) exit(2);

  // Create object to generate grasp candidates.
  CandidatesGenerator::Parameters generator_params;
  generator_params.num_samples_ = num_samples;
  generator_params.num_threads_ = num_threads;
  generator_params.plot_normals_ = plot_normals;
  generator_params.plot_grasps_ = plot_candidates;
  generator_params.remove_statistical_outliers_ = remove_outliers;
  generator_params.voxelize_ = false;
  generator_params.workspace_ = workspace;
  HandSearch::Parameters hand_search_params;
  hand_search_params.finger_width_ = finger_width;
  hand_search_params.hand_outer_diameter_ = hand_outer_diameter;
  hand_search_params.hand_depth_ = hand_depth;
  hand_search_params.hand_height_ = hand_height;
  hand_search_params.init_bite_ = init_bite;
  hand_search_params.nn_radius_frames_ = nn_radius;
  hand_search_params.num_orientations_ = num_orientations;
  hand_search_params.num_samples_ = num_samples;
  hand_search_params.num_threads_ = num_threads;
  hand_search_params.rotation_axis_ = rotation_axis;
  hand_search_params.friction_coeff_ = friction_coeff;
  hand_search_params.viable_thresh_ = viable_thresh;
  CandidatesGenerator candidates_generator(generator_params, hand_search_params);

  std::default_random_engine randn_generator;
  std::normal_distribution<double> randn(0.0, 1.0);

  // Set the camera pose.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_pose[3], camera_pose[6], camera_pose[9];

  // Create object to load point cloud from file.
  PointCloudRGB::Ptr mesh_cam_pts(new PointCloudRGB);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor; //filter to remove outliers
  sor.setStddevMulThresh (2.0);
  sor.setMeanK(30);
  int num_point_clouds = argc-(output_merged_pcd?5:4);
  for (int i=0; i<num_point_clouds; ++i) {
    PointCloudRGB::Ptr mesh_cam_to_merge = CloudCamera::loadPointCloudFromFile(argv[i+2]);
    // Remove outliers
    sor.setInputCloud (mesh_cam_to_merge);
    sor.filter (*mesh_cam_to_merge); 
    *mesh_cam_pts += *mesh_cam_to_merge;
  }

  if (voxelize) {
      std::cout << "Voxelizing point cloud..." << std::endl;
      pcl::VoxelGrid<pcl::PointXYZRGBA> grid;
      grid.setInputCloud(mesh_cam_pts);
      grid.setLeafSize(0.003f, 0.003f, 0.003f);
      grid.filter(*mesh_cam_pts);
  }

  // Add random noise to point cloud (for robustness testing)
  if (noise_std_cm>0) {
    for (int i=0; i<mesh_cam_pts->width; ++i) {
      // Generate a random unit vector. Sampled from unite ball
      double vx = randn(randn_generator);
      double vy = randn(randn_generator);
      double vz = randn(randn_generator);
      double  v = std::max(std::sqrt(vx*vx+vy*vy+vz*vz), 1e-10); // magnitude of the random vector
      double  a = randn(randn_generator) * noise_std_cm * 0.01; // in centimeter
      if (a<0) a = -a;
      // Multiply the magnitude of the noise on the unit vector
      vx = vx / v * a;
      vy = vy / v * a;
      vz = vz / v * a;
      // Add the noise to point cloud
      mesh_cam_pts->points[i].x += vx;
      mesh_cam_pts->points[i].y += vy;
      mesh_cam_pts->points[i].z += vz;
    }
  }

  Eigen::Affine3d mesh2cloud(loadRegistration(argv[num_point_clouds+2]));
  pcl::transformPointCloud(*mesh_cam_pts, *mesh_cam_pts, mesh2cloud);
  CloudCamera mesh_cam(mesh_cam_pts, 0, view_points);
  if (mesh_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "mesh_cam: Input point cloud is empty or does not exist!\n";
    return (-1);
  }
  
  std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();

  // Point cloud preprocessing: voxelize, remove statistical outliers, workspace filter, compute normals, subsample.
  candidates_generator.preprocessPointCloud(mesh_cam);

  // Generate a list of grasp candidates.
  std::vector<Grasp> candidates = candidates_generator.generateGraspCandidates(mesh_cam);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total => " << std::chrono::duration_cast<std::chrono::microseconds>(end - total_begin).count()*1e-6 << " seconds" << std::endl;

  std::ofstream output_fs(argv[num_point_clouds+3]);

  for (const auto &single_grasp : candidates) {
      std::vector<Grasp> grasp_vec, grasp_vec_centered;
      Eigen::Vector3d trans = single_grasp.getGraspBottom().cast<double>();
      Eigen::Vector3d top_vec = single_grasp.getGraspTop();
      Eigen::Matrix3d rot   = single_grasp.getFrame().cast<double>();
      double top = single_grasp.getTop();
      double bottom = single_grasp.getBottom();
      grasp_vec.push_back(single_grasp);
      
      Eigen::Matrix4d Trans_m, InvTrans_m;
      Eigen::Affine3d Trans, InvTrans;
      Eigen::Vector3d euler;
      if (rot(2,2)<0) // up-side-down
      {
        rot = rot * Eigen::AngleAxisd(PI, Eigen::Vector3d::UnitX());
      }
      euler = rot.eulerAngles(0, 1, 2); // roll, pitch, yaw
      double roll = euler(0);
      double pitch = euler(1);
      double yaw = euler(2);
      Trans_m.setIdentity();
      Trans_m.block<3,3>(0,0) = rot;
      Trans_m.block<3,1>(0,3) = trans;
      Trans.matrix() = Trans_m;
      InvTrans = Trans.inverse();
      InvTrans_m = InvTrans.matrix();

      if (roll<0) roll = std::min(PI_2, std::max(0.0, roll+PI_2));
      if (pitch<0) pitch = std::min(PI_2, std::max(0.0, pitch+PI_2));
      if (yaw<0) yaw = std::min(PI_2, std::max(0.0, yaw+PI_2));

      output_fs << std::setprecision(8) \
                << Trans_m.row(0)         << ' ' \
                << Trans_m.row(1)         << ' ' \
                << Trans_m.row(2)         << ' ' \
                << InvTrans_m.row(0)      << ' ' \
                << InvTrans_m.row(1)      << ' ' \
                << InvTrans_m.row(2)      << std::endl;
  }

  output_fs.close();

  if (output_merged_pcd) {
    pcl::io::savePCDFileBinary<pcl::PointXYZRGBA>(argv[num_point_clouds+4], *mesh_cam_pts);
  }
  
  return 0;
}
