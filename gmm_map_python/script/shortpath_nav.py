#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point,Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped
from gmm_map_python.msg import gmm
from gmm_map_python.msg import Submap
from gmm_map_python.msg import gmmFrame
from geometry_msgs.msg import Twist
from tf_conversions import transformations
import tf
import tf2_ros
from tf2_ros import TransformException
from tf2_sensor_msgs import do_transform_cloud
import rospy
import time
import numpy as np

import threading
import math
from generate_descriptor import Descriptor
from map_registration import Registrar

from TFGraph import TFGraph
from gmm_map_python.srv import *
import rospy
import tf
from geometry_msgs.msg import Twist
import sys, select, os
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion


WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.08
ANG_VEL_STEP_SIZE = 0.1


class Navigation:
    def __init__(self, self_id):
        self.self_robot_id = self_id
        self.self_constraint = [] #优化的单个机器人内部约束的
        self.new_self_submap = True # 如果有是新的关键帧,这个会变成 True,在 callback_self_pointcloud 中接收到新的关键帧率.并把他改成 False
        self.new_self_loop = False # 如果有是新的关键帧,这个会变成 True, 在后端优化中,讲这个置为false
        self.new_self_submap_count = 0 #调试过程中记录帧数,用于新建submap
        self.new_self_loop_count = 0 #调试过程中记录帧数,用于新建loop
        self.tf_listener = tf.TransformListener()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.transform_base_camera = TransformStamped()

        self.target = np.array([float(5),float(6)])
        self.dist = np.array([float(0),float(0)])

        self.registrar = Registrar()

        self.current_submap_id = 0
        self.list_of_submap = []
        self.baselink_frame = rospy.get_param('~baselink_frame','base_link')
        self.odom_frame = rospy.get_param('~odom_frame','odom')
        self.map_frame = rospy.get_param('~map_frame','map')

        self.br_ego_mapodom = tf.TransformBroadcaster()
        self.tans_ego_mapodom = TransformStamped()
        self.tans_ego_mapodom.header.frame_id = self.map_frame
        self.tans_ego_mapodom.child_frame_id = self.odom_frame
        self.tans_ego_mapodom.transform.rotation.w = 1

        self.path = Path()
        self.pathodom = Path()
        self.all_path = {}
        self.all_path['robot{}'.format(self.self_robot_id)] = self.path
        # self.octomap = octomap.OcTree(0.1)

        self.newsubmap_builder = None
        self.prefixsubmap_builder = None
        
        self.Descriptor = Descriptor(model_dir='/home/ubuntu/Downloads/model13.ckpt')
        self.descriptor_first = None


        self.path_pub = rospy.Publisher('path', Path, queue_size=1)
        self.path_pubodom = rospy.Publisher('pathodom', Path, queue_size=1)
        self.submap_publisher = rospy.Publisher('/all_submap',Submap, queue_size=1)
        self.all_path_pub = {}
        self.all_path_pub['robot{}'.format(self.self_robot_id)] = self.path_pub
        self.tmp_pub = rospy.Publisher('point_debug', PointCloud2,queue_size=1)
        self.test_pc2_pub = rospy.Publisher('all_localmap', PointCloud2,queue_size=1)
        self.GMM_pub = rospy.Publisher('GMMglobal', gmm,queue_size=1)
        self.GMMvisual_pub = rospy.Publisher('GMMvisual', MarkerArray, queue_size=10)
        self.ctr_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, self.callback_nav,queue_size=100)
        # self.self_pc_sub = rospy.Subscriber('sampled_points', PointCloud2, self.callback_nav,queue_size=1)
        print("start wait for service")

        rospy.wait_for_service('FeatureExtract/Feature_Service')
        try:
            self.feature_client=rospy.ServiceProxy('FeatureExtract/Feature_Service', FeatureExtraction)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        print("service init success!")

        self.transform_sub1_odom_br = tf2_ros.StaticTransformBroadcaster()

        #下面主要是为了处理其他的机器人的submap
        self.all_submap_lists = {}
        self.all_submap_locks = {}
        self.all_submap_lists['robot{}'.format(self.self_robot_id)] = self.list_of_submap
        self.all_submap_locks['robot{}'.format(self.self_robot_id)] = threading.Lock()

        self.tf_graph = TFGraph()
        self.tf_graph.new_tf_node(self.self_robot_id)
        print (self.tf_graph.graph_to_dot())

        self.match_thr = 0.4 #0.4for gmm_PR #0.4 for origin points
        self.fitness_thr =0.03 #0.04 #0.03

        # self.backpubglobalpoint = threading.Thread(target=self.pointmap_single_thread)
        # self.backpubglobalpoint.setDaemon(True)
        # self.backpubglobalpoint.start()
        
        self.transform_firstsubmap_odom = None

        #data transmission
        self.cloud_list=[]
        self.gmm_ori_list=[]
        self.gmm_after_list=[]

    

    def callback_new_self_pointcloud(self, data): #这个函数负责确定何时开始一个新的关键帧率,目前只是用来调试
        if (self.new_self_submap_count < 18):
            self.new_self_submap_count += 1
            # print(self.new_self_submap_count)
        else:
            if self.new_self_submap == False:
                self.new_self_submap = True # 当设置为True说明有一个新的关键帧
                self.new_self_submap_count = 0
            # print(self.new_self_submap)


    def callback_nav(self, data): #监听pointcloud,自己新建地图,并且保存对应的 odom.
        #submap 计数提取

        # self.callback_new_self_pointcloud(data)
       
        # assert isinstance(data, PointCloud2)
        # # print("PointCloud2 OK")
        # pointheader = data.header
        # pointtime = pointheader.stamp
        # if(pointtime < rospy.Time.now() - rospy.Duration(5) ):
        #     print(pointtime)
        #     print(rospy.Time.now() )
        #     return #这个主要是为了应对输入的点云太快的问题
        # # pointtime = rospy.Time(0)
        # # print(self.baselink_frame)

        # try:
        #     self.tf_listener.waitForTransform(self.baselink_frame,pointheader.frame_id,pointtime,rospy.Duration(0.11))
        #     self.transform_base_camera = self.tf2_buffer.lookup_transform(self.baselink_frame,pointheader.frame_id, pointtime) #将点云转换到当前 base_link 坐标系的变换(一般来说是固定的), 查询出来的是, source 坐标系在 target 坐标系中的位置.
        # except TransformException:
        #     print("self.baselink_frame:",self.baselink_frame)
        #     print("pointheader.frame_id:",pointheader.frame_id)
        #     print ('no tf')
        # baselink_pointcloud = do_transform_cloud(data,self.transform_base_camera) #将 source(camera) 的点云 变换到 taget(base_link) 坐标系

        try:
            self.tf_listener.waitForTransform(self.odom_frame,self.baselink_frame,rospy.Time(),rospy.Duration(5))
            transform_odom_base = self.tf2_buffer.lookup_transform(self.odom_frame,self.baselink_frame, rospy.Time()) #得到了 baselink 在odom 坐标系中的位置
            #print(transform_odom_base)
        except TransformException:
            print("self.odom_frame:",self.odom_frame)
            print("self.baselink_frame",self.baselink_frame)
            print ('no tf -- break')
            return
        # print("Before new Submap!",self.new_self_submap)

        # print("self.map_frame", self.map_frame)
        # print("self.baselink_frame", self.baselink_frame)       
        try:
            self.tf_listener.waitForTransform(self.map_frame, self.baselink_frame, rospy.Time(), rospy.Duration(5))
            #等待Duration=5s，判断map是否转换为base_link
            #transform_map_base = TransformStamped()

            # transform_map_base = self.tf2_buffer.lookup_transform(self.map_frame, self.baselink_frame, pointtime) # baselink 在 map 中的位置。
            # print(transform_map_base)
            (pos,ori) = self.tf_listener.lookupTransform(self.map_frame, self.baselink_frame,rospy.Duration(0.0))

            print("pos_x:",pos[0])
            print("pos_y:",pos[1])
            print("pos_z:",pos[2])
            print("ori_x:",ori[0])
            print("ori_y:",ori[1])
            print("ori_z:",ori[2])
            print("ori_w:",ori[3])

            #(trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            #rospy_Time(0)指最近时刻存储的数据
            #得到从 map 到 base_link 的变换，在实际使用时，转换得出的坐标是在 base_link 坐标系下的。
            euler = transformations.euler_from_quaternion(ori)    #将四元数转换为欧拉角

            # th = euler[2] / pi * 180
            # r = rospy.Rate(100)    #设置速率，每秒发100次
            # r.sleep()
            # while not rospy.is_shutdown():    #如果节点已经关闭则is_shutdown()函数返回一个True，反之返回False
            #     print (x, y, th)    #输出坐标。
            #     r.sleep() 

            turtlebot3_model = rospy.get_param("model", "waffle")

            target_linear_vel   = float(0.0)
            target_angular_vel  = float(0.0)
            control_linear_vel  = float(0.0)
            control_angular_vel = float(0.0)

            #print(msg)
            # while not rospy.is_shutdown():
            self.dist[0] = float(float(self.target[0])-float(pos[0]))
            self.dist[1] = float(float(self.target[1])-float(pos[1]))
            print(self.dist[0],self.dist[1])




            if abs(self.dist[0])<0.2 and abs(self.dist[0])<0.2:
                target_linear_vel   = 0.0
                control_linear_vel  = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0  
                print(self.vels(target_linear_vel, target_angular_vel))
            elif self.dist[0]>0 :
                print("on your right")
                theta = math.atan(self.dist[1]/self.dist[0])
                theta = math.degrees(theta)
                print("euler",math.degrees(euler[2]))
                print("theta",theta)
                if theta - math.degrees(euler[2]) >5:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = target_angular_vel + ANG_VEL_STEP_SIZE
                    print(self.vels(target_linear_vel,target_angular_vel))
                elif theta - math.degrees(euler[2])<-5:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = target_angular_vel - ANG_VEL_STEP_SIZE
                    print(self.vels(target_linear_vel,target_angular_vel))
                else:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = 0
                    print(self.vels(target_linear_vel,target_angular_vel))                    
            elif self.dist[0]<0 :
                print("on your left")
                if self.dist[1]>0:
                    theta = math.atan(self.dist[1]/self.dist[0])
                    theta = math.degrees(theta) +180
                if self.dist[1]<0:
                    theta = math.atan(self.dist[1]/self.dist[0])
                    theta = math.degrees(theta) -180
                print("euler",math.degrees(euler[2]))
                print("theta",theta)
                if -5 <= (theta - math.degrees(euler[2])) <=5:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = 0
                    print(self.vels(target_linear_vel,target_angular_vel))
                elif theta - math.degrees(euler[2])< -5:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = target_angular_vel - ANG_VEL_STEP_SIZE
                    print(self.vels(target_linear_vel,target_angular_vel))
                elif theta - math.degrees(euler[2])> 5:
                    target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                    target_angular_vel = target_angular_vel + ANG_VEL_STEP_SIZE
                    print(self.vels(target_linear_vel,target_angular_vel))
            elif self.dist[0]==0 :
                if self.dist[1]>0:
                    if math.degrees(euler[2])!=90:
                        target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                        target_angular_vel = target_angular_vel + ANG_VEL_STEP_SIZE
                    else:
                        target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                        target_angular_vel = 0
                        print(self.vels(target_linear_vel,target_angular_vel))
                if self.dist[1]<0:
                    if math.degrees(euler[2])!=-90:
                        target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                        target_angular_vel = target_angular_vel + ANG_VEL_STEP_SIZE
                    else:
                        target_linear_vel = target_linear_vel + LIN_VEL_STEP_SIZE
                        target_angular_vel = 0
                        print(self.vels(target_linear_vel,target_angular_vel))


          
            twist = Twist()
            
            control_linear_vel = target_linear_vel
            twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0
            

            control_angular_vel = target_angular_vel
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel
            

            self.ctr_pub.publish(twist)


        except TransformException:
            rospy.loginfo("tf Error")
        #try ... except ... 进行异常处理，若 try... 出现异常，则将错误直接输出打印，而不是以报错的形式显示。

  
        # finally:
        #     twist = Twist()
        #     twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        #     twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        #     self.ctr_pub.publish(twist)


    def new_SPHERE_Marker(self, frame_id, position, scale, id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id  # enumerate subsequent markers here
        marker.action = Marker.ADD  # can be ADD, REMOVE, or MODIFY
        marker.ns = "vehicle_model"
        marker.type = Marker.SPHERE
        
        marker.pose.position = position
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale = scale

        marker.color.r = 1.0
        marker.color.g = 0.50
        marker.color.b = 0.0
        marker.color.a = 1.0 
        return marker


    def GMMvisual_cb(self,data):
        #把代码从visualgmm.py粘过来，创建一个发可视化的topic
        #print("start GMM visualization")
        markerarray = MarkerArray()
        cu_gmm = data
        cu_mix_num = cu_gmm.mix_num
        for jndex in range(cu_mix_num):
            marker = self.new_SPHERE_Marker(self.odom_frame,Point(cu_gmm.x[jndex],cu_gmm.y[jndex],cu_gmm.z[jndex]),
            Vector3(10*cu_gmm.x_var[jndex],10*cu_gmm.y_var[jndex],10*cu_gmm.z_var[jndex]),
            jndex )
            if cu_gmm.z[jndex]>0.1:
                markerarray.markers.append(marker)
        # print(markerarray)
        self.GMMvisual_pub.publish(markerarray)


    def vels(self,target_linear_vel, target_angular_vel):
        return "currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel)




def main():
    rospy.init_node('nav', anonymous=True)
    Robot1 = Navigation(rospy.get_param('~robot_id',1))
    rospy.spin()

if __name__ == "__main__":
    main()
