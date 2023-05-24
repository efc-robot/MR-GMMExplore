#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include<tf/transform_listener.h>
#include <ros/duration.h>
#include<thread>
#include<mutex>
#include "math.h"
#include <climits>// give variable the biggest number
#include "time.h"

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>	
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include<Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <tf/transform_broadcaster.h>

#include "submap/KMeans.h"
#include "submap/GMM.h"
#include <submap/gmm.h> //gmm_msg
#include <submap/gmmlist.h> //gmmlist_msg

class Submap {
public:
  Submap() {  }
  ~Submap() {}
void initMap(ros::NodeHandle nh);
void pubSubinfo();
void Mapmerge();
void GMM_training(int novel_submap_frame);
void mapCallback(const sensor_msgs::PointCloud2 img);
Eigen::Matrix4f  TransformToMatrix(const tf::StampedTransform& transform) ;
void Global_Pointcloud_Publisher();
void Global_GMM_Publisher();
void Submap_GMM_building();
void Loop_Detection();
void G2G_merging(int newframe);
void generate_path();
bool loop_detection(); //no subcribe, only single agent loop detection
void back_end_optimazation();

//G2G_merge
double CalDistance(int dim,double* mean1, double*var1, double* mean2, double*var2);
double trace(int dim, double* var);
double* MM(int dim, double* var1, double* var2);
double* Madd(int dim, double* var1, double* var2, double* var3);


private:
    // ros::NodeHandle nh_;
    std::vector<sensor_msgs::PointCloud2> Submap_list_;
    std::vector<tf::StampedTransform> SubTF_list_; //pose_list will be optimized after backend optimazation
    std::vector<geometry_msgs::Pose> Trans_origin_;
    std::vector<GMM*> SubGMM_list_; // all the gmm submap are in the global TF, not local
    sensor_msgs::PointCloud2 Globalmap_; 
    GMM *GlobalGMM_;
    pcl::PointCloud<pcl::PointXYZ> global_cloud_;
    int mapcnt_;// extract a Submap every mapcnt frames, to be improved as feature-based extractor
    int submap_num_;
    int subgmm_num_;
    int gmm_unmerge_;
    ros::Publisher gobalmap_pub_; 
    ros::Publisher gmm_pub_; 
    ros::Publisher path_pub_;
    ros::Publisher gmm_list_pub_;
    // ros::Publisher submap_pub_; 
    // ros::Publisher subTF_pub_;
    ros::Subscriber map_sub_;
    tf::TransformListener TFlistener_;
    std::mutex Globalmap_mutex_;
    double dis_threshold_;
    const int dim_=3;
    const int cluster_num_=100;
    nav_msgs::Path path_; 
    ros::Time current_time_;
    submap::gmm gmm_msg_; //current gmm message
    submap::gmmlist gmm_list_msg_; //overall gmm list message


};