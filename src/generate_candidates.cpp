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
  std::cout << s << std::endl;
  Eigen::Matrix4d mesh2cloud;
  mesh2cloud.row(3) << 0,0,0,1;
  std::stringstream ss;
  ss << s;
  std::string substr;
  for (int i=0; i<12; ++i) {
      std::getline(ss, substr, ',');
      std::cout << i << ' ' << substr << std::endl;
      mesh2cloud.col(i/3).row(i%3) << std::stod(substr);
  }
  return Eigen::Affine3d(mesh2cloud);
}

Grasp transformGrasp(const Grasp& grasp, const FingerHand &finger_hand, Eigen::Affine3d transform)
{
  Eigen::Matrix3d I;
  I.setIdentity();
  Grasp grasp_I(transform*grasp.getSample(), I, finger_hand);
  grasp_I.setGraspWidth(grasp.getGraspWidth());
  grasp_I.setGraspBottom(transform*grasp.getGraspBottom());
  grasp_I.setGraspTop(transform*grasp.getGraspTop());
  grasp_I.setGraspSurface(transform*grasp.getGraspSurface());
  return grasp_I;
}

inline double random_uniform(void) 
{
  return std::max(1e-8, std::min(1.0-1e-8,(double)rand() / (double)RAND_MAX));
}

inline Eigen::Quaterniond randomQuaternion_sample(double u1, double u2, double u3)
{
  Eigen::Quaterniond random_quat(std::sqrt(1.-u1)*std::sin(PI_2*u2), std::sqrt(1.-u1)*std::cos(PI_2*u2), \
                                 std::sqrt(u1   )*std::sin(PI_2*u3), std::sqrt(u1   )*std::cos(PI_2*u3));
  return random_quat;
}

Eigen::Quaterniond randomQuaternion(void) 
{
  // Random Rotation
  double u1 = random_uniform();
  double u2 = random_uniform();
  double u3 = random_uniform();
  return randomQuaternion_sample(u1, u2, u3);
}


int main(int argc, char* argv[])
{
  srand(time(NULL));
  // Read parameters from configuration file.
  ConfigFile config_file(argv[1]);

  double finger_width = config_file.getValueOfKey<double>("finger_width", 0.01);
  double hand_outer_diameter  = config_file.getValueOfKey<double>("hand_outer_diameter", 0.12);
  double hand_depth = config_file.getValueOfKey<double>("hand_depth", 0.06);
  double hand_height  = config_file.getValueOfKey<double>("hand_height", 0.02);
  double init_bite  = config_file.getValueOfKey<double>("init_bite", 0.01);

  std::cout << "finger_width: " << finger_width << "\n";
  std::cout << "hand_outer_diameter: " << hand_outer_diameter << "\n";
  std::cout << "hand_depth: " << hand_depth << "\n";
  std::cout << "hand_height: " << hand_height << "\n";
  std::cout << "init_bite: " << init_bite << "\n";

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

  bool plot_grasps = config_file.getValueOfKey<bool>("plot_grasps", true);
  bool plot_candidates = config_file.getValueOfKey<bool>("plot_candidates", true);
  bool plot_normals = config_file.getValueOfKey<bool>("plot_normals", false);
  int max_samples = config_file.getValueOfKey<int>("max_samples", 10);
  std::cout << "plot_grasps: " << plot_grasps << "\n";
  std::cout << "plot_candidates: " << plot_candidates << "\n";
  std::cout << "plot_normals: " << plot_normals << "\n";
  std::cout << "max_samples: " << max_samples << "\n";

  // Create object to generate grasp candidates.
  CandidatesGenerator::Parameters generator_params;
  generator_params.num_samples_ = num_samples;
  generator_params.num_threads_ = num_threads;
  generator_params.plot_normals_ = plot_normals;
  generator_params.plot_grasps_ = plot_candidates;
  generator_params.remove_statistical_outliers_ = remove_outliers;
  generator_params.voxelize_ = voxelize;
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
  CandidatesGenerator candidates_generator(generator_params, hand_search_params);
  HandSearch handsearch(hand_search_params);

  // Set the camera pose.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_pose[3], camera_pose[6], camera_pose[9];

  // Random Rotation
  /*
  for (double u1=1e-8; u1<1-1e-8; u1+=0.05)
    for (double u2=1e-8; u2<1-1e-8; u2+=0.05)
      for (double u3=1e-8; u3<1-1e-8; u3+=0.05) 
      {
        Eigen::Quaterniond random_quat(randomQuaternion());
        std::cout << "Random Euler Angle: " << random_quat.toRotationMatrix().eulerAngles(0, 1, 2).transpose() << std::endl;
      }
  */

  Eigen::Affine3d mesh2cloud(loadRegistration(argv[4]));
  std::cout << mesh2cloud.matrix() << std::endl;
  // pcl::transformPointCloud(*mesh_cam.getCloudOriginal(), *new_pc, InvTrans);

  // Create object to load point cloud from file.
  CloudCamera cloud_cam(argv[3], view_points);
  PointCloudRGB::Ptr mesh_cam_pts = CloudCamera::loadPointCloudFromFile(argv[2]);
  pcl::transformPointCloud(*mesh_cam_pts, *mesh_cam_pts, mesh2cloud);
  CloudCamera mesh_cam(mesh_cam_pts, 0, view_points);
  if (mesh_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "mesh_cam: Input point cloud is empty or does not exist!\n";
    return (-1);
  }
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "cloud_cam: Input point cloud is empty or does not exist!\n";
    return (-1);
  }
  std::chrono::steady_clock::time_point total_begin = std::chrono::steady_clock::now();

  // Point cloud preprocessing: voxelize, remove statistical outliers, workspace filter, compute normals, subsample.
  candidates_generator.preprocessPointCloud(mesh_cam);

  // Generate a list of grasp candidates.
  std::vector<Grasp> candidates = candidates_generator.generateGraspCandidates(mesh_cam);
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::vector<int> labels = handsearch.reevaluateHypotheses(mesh_cam, candidates);
  std::vector<Grasp> good_grasps;
  std::vector<int> good_index;
  for (int i=0; i<labels.size(); ++i) {
      if (labels[i]==2) good_index.push_back(i); // good_grasps.push_back(candidates[i]);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Evaluation => " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()*1e-6 << " seconds" << std::endl;
  std::cout << "Total => " << std::chrono::duration_cast<std::chrono::microseconds>(end - total_begin).count()*1e-6 << " seconds" << std::endl;

  std::random_shuffle(good_index.begin(), good_index.end());
  
  for (int i=0; i<std::min(max_samples, (int)good_index.size()); ++i) {
      good_grasps.push_back(candidates[good_index[i]]);
  }

  std::cout << "Generated " << good_index.size() << " good grasps." << std::endl;
  std::cout << "Selected " << good_grasps.size() << " good grasps." << std::endl;
  if (plot_grasps) {
      Plot plotter;
      plotter.plotFingers3D(good_grasps, mesh_cam.getCloudOriginal(), "Good Grasps", hand_search_params.hand_outer_diameter_, 
              hand_search_params.finger_width_, hand_search_params.hand_depth_, hand_search_params.hand_height_);
  }
  // Eigen::Vector3d euler_mean, euler2_mean;
  std::vector<int> roll_h(32,0), pitch_h(32,0), yaw_h(32,0);
  // euler_mean.setZero();
  // euler2_mean.setZero();
  for (int i=0; i<good_grasps.size(); ++i) {
      Grasp single_grasp = good_grasps[i];
      std::vector<Grasp> grasp_vec, grasp_vec_centered;
      Eigen::Vector3d trans = single_grasp.getGraspBottom().cast<double>();
      Eigen::Vector3d top_vec = single_grasp.getGraspTop();
      Eigen::Matrix3d rot   = single_grasp.getFrame().cast<double>();
      double top = single_grasp.getTop();
      double bottom = single_grasp.getBottom();
      grasp_vec.push_back(single_grasp);
      if (plot_grasps) {
          Plot plotter;
          plotter.plotFingers3D(grasp_vec, mesh_cam.getCloudOriginal(), "Good Grasps (mesh)", hand_search_params.hand_outer_diameter_, 
                  hand_search_params.finger_width_, hand_search_params.hand_depth_, hand_search_params.hand_height_);
      } 
      if (plot_grasps) {
          Plot plotter;
          plotter.plotFingers3D(grasp_vec, cloud_cam.getCloudOriginal(), "Good Grasps (cloud)", hand_search_params.hand_outer_diameter_, 
                  hand_search_params.finger_width_, hand_search_params.hand_depth_, hand_search_params.hand_height_);
      }
      Eigen::Matrix4d Trans_m;
      Eigen::Affine3d Trans, InvTrans;
      Eigen::Vector3d euler;
      if (rot(2,2)<0) // up-side-down
      {
        rot = rot * Eigen::AngleAxisd(PI, Eigen::Vector3d::UnitX());
      }
      euler = rot.eulerAngles(0, 1, 2); // roll, pitch, yaw
      Trans_m.setIdentity();
      Trans_m.block<3,3>(0,0) = rot;
      Trans_m.block<3,1>(0,3) = trans;
      Trans.matrix() = Trans_m;
      InvTrans = Trans.inverse();
      double roll = euler(0);
      double pitch = euler(1);
      double yaw = euler(2);

      std::cout << "Euler Angles (x-roll, y-pitch, z-yaw): " << euler.transpose() << std::endl;
      std::cout << "Translation: " << trans.transpose() << std::endl;
      // euler_mean += euler;
      // euler2_mean += (Eigen::Vector3d)(euler.array()*euler.array());

      if (roll<0) roll = std::min(PI_2, std::max(0.0, roll+PI_2));
      if (pitch<0) pitch = std::min(PI_2, std::max(0.0, pitch+PI_2));
      if (yaw<0) yaw = std::min(PI_2, std::max(0.0, yaw+PI_2));
      ++roll_h[std::min((int)roll_h.size()-1,(int)(roll/PI_2*roll_h.size()))];
      ++pitch_h[std::min((int)pitch_h.size()-1,(int)(pitch/PI_2*pitch_h.size()))];
      ++yaw_h[std::min((int)yaw_h.size()-1,(int)(yaw/PI_2*yaw_h.size()))];

      /*
      std::cout << Trans.matrix()    << std::endl \
                << InvTrans.matrix() << std::endl;

      std::cout << std::endl;

      std::cout << top     << std::endl \
                << top_vec << std::endl \
                << std::endl \
                << bottom  << std::endl \
                << trans   << std::endl;

      std::cout << std::endl;
      */

      std::vector<int> enclosed_indecies;
      FingerHand finger_hand(finger_width, hand_outer_diameter, hand_depth);
      Grasp single_grasp_I = transformGrasp(single_grasp, finger_hand, InvTrans);
      grasp_vec_centered.push_back(single_grasp_I);

      pcl::CropBox<pcl::PointXYZRGBA> boxFilter(true);
      boxFilter.setMin(Eigen::Vector4f(0, -(hand_outer_diameter-finger_width*2)/2.0, -hand_height/2.0, 1.0));
      boxFilter.setMax(Eigen::Vector4f(top-bottom, (hand_outer_diameter-finger_width*2)/2.0, hand_height/2.0, 1.0));
      boxFilter.setTransform(InvTrans.cast<float>());

      PointCloudRGB::Ptr new_pc(new PointCloudRGB), new_pc_crop(new PointCloudRGB);
      boxFilter.setInputCloud(cloud_cam.getCloudOriginal());
      boxFilter.filter(*new_pc_crop);
      boxFilter.filter(enclosed_indecies);
      // for (auto v : enclosed_indecies) std::cout << v << ' ';
      // std::cout << std::endl;
      pcl::transformPointCloud(*new_pc_crop, *new_pc_crop, InvTrans);
      pcl::transformPointCloud(*cloud_cam.getCloudOriginal(), *new_pc, InvTrans);
      if (plot_grasps) {
          Plot plotter;
          plotter.plotFingers3D(grasp_vec_centered, new_pc, "Good Grasps (centered)", hand_search_params.hand_outer_diameter_, 
                  hand_search_params.finger_width_, hand_search_params.hand_depth_, hand_search_params.hand_height_);
      }

      if (plot_grasps) {
          Plot plotter;
          plotter.plotFingers3D(grasp_vec_centered, new_pc_crop, "Good Grasps (centered-cropped)", hand_search_params.hand_outer_diameter_, 
                  hand_search_params.finger_width_, hand_search_params.hand_depth_, hand_search_params.hand_height_);
      }
      new_pc->clear();
      new_pc_crop->clear();
  }

  // euler_mean /= (double)good_grasps.size();
  // euler2_mean /= (double)good_grasps.size();

  // std::cout << "Euler Means: " << euler_mean.transpose() << std::endl;
  // std::cout << "Euler Vars : " << (euler2_mean-(Eigen::Vector3d)(euler_mean.array()*euler_mean.array())).transpose() << std::endl;

  std::cout << "Roll histogram (0~2*pi):" << std::endl;
  for (auto v : roll_h) std::cout << v << ' ';
  std::cout << std::endl;

  std::cout << "Pitch histogram (0~2*pi):" << std::endl;
  for (auto v : pitch_h) std::cout << v << ' ';
  std::cout << std::endl;

  std::cout << "Yaw histogram (0~2*pi):" << std::endl;
  for (auto v : yaw_h) std::cout << v << ' ';
  std::cout << std::endl;
  
  return 0;
}
