#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from gmm_map_python import msg
import rospy
from tf_conversions import transformations
from math import pi
import tf
import tf2_ros
from tf2_ros import TransformException
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point,Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped
from gmm_map_python.msg import gmm, gmmFrame
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
from sensor_msgs.msg import PointCloud2

from GMMmap import GMMFrame

from visualgmm import new_SPHERE_Marker
from visualgmm import subgmm_cb

from MapBuilderNode import InsubmapProcess
from MapBuilderNode import np2pointcloud2
from MapBuilderNode import TrajMapBuilder

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from numpy.linalg import inv, det

class Robot:
    def __init__(self, self_id):
        self.self_robot_id = self_id
        self.tf_listener = tf.TransformListener()    
        #创建了监听器，它通过线路接收tf转换，并将其缓冲10s。若用C++写，需设置等待10s缓冲。
        self.tf2_buffer = tf2_ros.Buffer()
        self.baselink_frame = rospy.get_param('~baselink_frame','base_link')
        self.odom_frame = rospy.get_param('~odom_frame','odom')
        self.map_frame = rospy.get_param('~map_frame','map')
        self.GMMvisual_pub = rospy.Publisher('GMMvisual', MarkerArray, queue_size=10)
        self.GMM_pub = rospy.Publisher('GMMglobal', gmm,queue_size=1)
        self.test_pc2_pub = rospy.Publisher('all_localmap', PointCloud2,queue_size=1)
        self.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, self.GMMmodel_cb,queue_size=100)
        self.self_pc_sub = rospy.Subscriber('sampled_points', PointCloud2, self.get_pos,queue_size=1)
        #self.GMMmodel_pub = rospy.Publisher('GMMmodel', MarkerArray, queue_size=10)
        

    def get_pos(self,data):
        assert isinstance(data, PointCloud2)
        # print("PointCloud2 OK")
        pointheader = data.header
        pointtime = pointheader.stamp
        try:
            self.tf_listener.waitForTransform(self.map_frame, self.baselink_frame, pointtime, rospy.Duration(5))
            #等待Duration=0.11s，判断map是否转换为base_link
            transform_map_base = self.tf2_buffer.lookup_transform(self.map_frame, self.baselink_frame, pointtime) # baselink 在 map 中的位置。
            print(transform_map_base)
            #(trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            #(trans, rot) = self.transform_map_base
            #rospy_Time(0)指最近时刻存储的数据
            #得到从 map 到 base_link 的变换，在实际使用时，转换得出的坐标是在 base_link 坐标系下的。
        except TransformException:
            rospy.loginfo("tf Error")
            #return None
        #try ... except ... 进行异常处理，若 try... 出现异常，则将错误直接输出打印，而不是以报错的形式显示。
        # euler = transformations.euler_from_quaternion(rot)    #将四元数转换为欧拉角
        # x = trans[0]
        # y = trans[1]
        # th = euler[2] / pi * 180
        # r = rospy.Rate(100)    #设置速率，每秒发100次
        # r.sleep()
        # while not rospy.is_shutdown():    #如果节点已经关闭则is_shutdown()函数返回一个True，反之返回False
        #     print (x, y, th)    #输出坐标。
        #     r.sleep()  
        # return (x, y, th)

    def GMMmodel_cb(self,data):
        markerarray = MarkerArray()
        cu_gmm = data
        cu_mix_num = cu_gmm.mix_num
        for jndex in range(cu_mix_num):
            marker = new_SPHERE_Marker(self.odom_frame,Point(cu_gmm.x[jndex],cu_gmm.y[jndex],cu_gmm.z[jndex]),
            Vector3(10*cu_gmm.x_var[jndex],10*cu_gmm.y_var[jndex],10*cu_gmm.z_var[jndex]),
            jndex )
            markerarray.markers.append(marker)
        # print(markerarray)
        self.GMMvisual_pub.publish(markerarray) 

        # assert isinstance(data, PointCloud2)
        # # print("PointCloud2 OK")
        # pointheader = data.header
        # pointtime = pointheader.stamp
        # try:
        #     self.tf_listener.waitForTransform("robot1/map", self.baselink_frame, pointtime, rospy.Duration(0.11))
        #     #等待Duration=0.11s，判断map是否转换为base_link
        #     self.transform_map_base = self.tf2_buffer.lookup_transform("robot1/map", self.baselink_frame, pointtime) # baselink 在 map 中的位置。
        #     #(trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        #     (trans, rot) = self.transform_map_base
        #     #rospy_Time(0)指最近时刻存储的数据
        #     #得到从 map 到 base_link 的变换，在实际使用时，转换得出的坐标是在 base_link 坐标系下的。
        # except TransformException:
        #     rospy.loginfo("tf Error")
        #     return None
        # #try ... except ... 进行异常处理，若 try... 出现异常，则将错误直接输出打印，而不是以报错的形式显示。
        # euler = transformations.euler_from_quaternion(rot)    #将四元数转换为欧拉角
        # x = trans[0]
        # y = trans[1]
        # th = euler[2] / pi * 180
        # r = rospy.Rate(100)    #设置速率，每秒发100次
        # r.sleep()
        # while not rospy.is_shutdown():    #如果节点已经关闭则is_shutdown()函数返回一个True，反之返回False
        #     print (x, y, th)    #输出坐标。
        #     r.sleep()  


        # Pi = np.array(data.submap_gmm.weights)
        # mu = np.array(data.submap_gmm.means).reshape(-1,3)
        # Sigma = np.array(data.submap_gmm.covariances).reshape(-1,3)
        # cu_gmm = data
        # cu_mix_num = cu_gmm.mix_num
        # for jndex in range(cu_mix_num):

        #     x = cu_gmm.x[jndex]
        #     y = cu_gmm.y[jndex]
        #     z = cu_gmm.z[jndex]
        #     fig = plt.figure()
        #     # 绘制3d图形
        #     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        #     ax1.plot_surface(x, y, z)
        #     # 绘制等高线
        #     ax2 = fig.add_subplot(1, 2, 2)
        #     ax2.contour(x, y, z)

        #     plt.show()   

    def gaussion_mixture(self, x, Pi, mu, Sigma):
        dim = len(x)
        constant = (2*np.pi)**(-dim/2) * det(Sigma)** (-0.5)
        z = 0
        for idx in range(len(Pi)):
            z += Pi[idx] * constant*np.exp(-0.5*(x-mu).dot(inv(Sigma)).dot(x-mu))
        return z

    def pointmap_merge_single(self): #将每一帧的点云都合并在一起可视化出来
        # final_result = PointCloud2()
        tmpmap_builder = InsubmapProcess(0,0,Pose(),Pose(),self.Descriptor,add_time=None, point_clouds=np.zeros((0,3)), init_gmm=GMMFrame(), freezed=False, descriptor=None, feature_point=np.zeros((0,3)), feature_gmm=GMMFrame())
        # print(tmpmap_builder.submap_gmm.means.shape)
        # if (len(self.list_of_submap) <= 1):
        #     return 
        # print("self.all_submap_lists.items():",self.all_submap_lists.items())
        for robot_id, submap_list in self.all_submap_lists.items():
            # print 'pointmap_thread: {}'.format(robot_id)
            tf_inter_robot = None
            if robot_id != 'robot{}'.format(self.self_robot_id):
                tf_inter_robot = self.tf_graph.get_tf(int(robot_id[5:]), self.self_robot_id)
                if tf_inter_robot == None:
                    continue

            for submapinst in submap_list:
                # print 'pointmap_thread: {} - {}'.format(robot_id,submapinst.submap_index)
                [showpoints, point_debug] = submapinst.pointmap2odom(tf_inter_robot) #不同的点云都在map坐标系中  
                # self.tmp_pub.publish(point_debug)
                # print("tf_inter_robot:",tf_inter_robot)            
                submapinst_tmp=submapinst.gmmmap2odom(tf_inter_robot)#GMM坐标系转换，因为gmmmap2odom会改自身的参数，所以要新开一个变量暂存
                # print("gmm2odom:",submapinst.submap_gmm)
                tmpmap_builder.insert_point(showpoints, ifcheck=False, if_filter=True, ifgmm=False) #GMMmap也会在此函数中添加
                tmpmap_builder.GMMmerge_global(submapinst_tmp) #GMM参数添加
        
        showpoints = np2pointcloud2(tmpmap_builder.submap_point_clouds,self.map_frame)
        odom_pose=Pose()
        showGMMs= tmpmap_builder.submap_gmm.GMMmsg(self.map_frame,odom_pose) #showGMMs是GMM消息
        if showGMMs.mix_num!=0:
            # print("gmmmsg:",showGMMs) 
            # print("mix_num:",tmpmap_builder.submap_gmm.mix_num)
            self.GMM_pub.publish(showGMMs) #先将GMM消息发出去，由visualgmm.py发送可视化     
        # self.refresh_ego_mapodom_tf() #更新map和pose的tf树
        # self.refresh_inter_robot_tf()
        self.test_pc2_pub.publish(showpoints) # 调试点云的时候的可视化     

    def pointmap_single_thread(self): #可视化线程——发布点云+GMM消息
        while (1):
            time.sleep(2)
            # print("point+GMM_thread")
            self.pointmap_merge_single()

    # def pos_discard(self,data):
    #     cu_gmm = data
    #     cu_mix_num = cu_gmm.mix_num
    #     for jndex in range(cu_mix_num):

def main():
    rospy.init_node('get_pos_demo',anonymous=True)   
        #启动节点get_pos_demo, 同时为节点命名， 若anonymous为真则节点会自动补充名字，实际名字以get_pos_demo_12345等表示
        #若为假，则系统不会补充名字，采用用户命名。如果有重名，则最新启动的节点会注销掉之前的同名节点。
    Robot.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, Robot.GMMmodel_cb,queue_size=100)
    #Robot.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, Robot.pos_discard,queue_size=100)

    # r = rospy.Rate(100)    #设置速率，每秒发100次
    # r.sleep()
    # while not rospy.is_shutdown():    #如果节点已经关闭则is_shutdown()函数返回一个True，反之返回False
    #     print (Robot.get_pos())    #输出坐标。
    #     r.sleep()        
    Robot1 = Robot(rospy.get_param('~robot_id',1))
    rospy.spin()

if __name__ == "__main__":
    main()