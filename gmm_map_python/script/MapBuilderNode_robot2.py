#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ast import Num
from audioop import mul
from cmath import pi
from itertools import count
from numbers import Rational
from operator import sub
import sys
import xlwt
import scipy.stats as st
from std_msgs.msg import Header, String,Float32MultiArray,Float32
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
from nav_msgs.msg import Path
import submap
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
import math
import random
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
import random
from GMMmap import GMMFrame
from datetime import datetime
from autolab_core import RigidTransform
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion

LIN_VEL_STEP_SIZE = 0.30
ANG_VEL_STEP_SIZE = 0.18

class Navigation:
    def __init__(self, self_id):
        print("yessssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
        self.self_robot_id = self_id
        self.tf_listener = tf.TransformListener()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.transform_base_camera = TransformStamped()
        self.baselink_frame = rospy.get_param('~baselink_frame','base_link')
        self.odom_frame = rospy.get_param('~odom_frame','odom')
        self.map_frame = rospy.get_param('~map_frame','map')
        self.ctr_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        
        self.pub_location = rospy.Publisher('/location',Float32MultiArray,queue_size=10)
        self.sub_state = rospy.Subscriber('state',Float32MultiArray,self.other_location,queue_size=10)
        self.sub_loc = rospy.Subscriber('/location',Float32MultiArray,self.mr_target,queue_size=10)
        self.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, self.callback_nav,queue_size=10)
        self.sub_subgmmtrans = rospy.Subscriber('/all_submap', Submap, self.callback, queue_size=10)
        self.cov_time = rospy.Publisher('coverage2',Float32MultiArray,queue_size=10)
        self.length_time = rospy.Publisher('length2',Float32MultiArray,queue_size=10)
        self.pub_pos = rospy.Publisher('current_pos',Float32MultiArray,queue_size=10)
        self.pub_target = rospy.Publisher('target_pos',Float32MultiArray,queue_size=10)
        self.sub_target = rospy.Subscriber('target_pos',Float32MultiArray,self.gmm_nav,queue_size=10)
        self.pub = rospy.Publisher('gmm_after_trans', gmm, queue_size=10)
        self.gmm_sub = rospy.Subscriber('gmm_after_trans',gmm, self.subgmm_cb,queue_size=100)
        
        self.W = 16 
        self.L = 24 
        self.co_explore = 0
        self.begin = datetime.now()
        self.coverage_time = xlwt.Workbook(encoding='utf-8')
        self.excelsheet = self.coverage_time.add_sheet('sheet1')
        self.coverage_time.save('cov_time.xls')
        self.gmm_map = np.zeros((self.W,self.L))
        self.step_sum = 0
        self.target_list = list()
        self.target_remove = list()
        self.gmm_ground = list()
        self.gmm_ground_all = list()
        self.gmm_wall_up = list()
        self.gmm_wall_down = list()
        self.gmm_wall = list()
        self.pos = 0
        self.ori = 0
        self.euler = 0
        self.t = 0
        self.other_target = np.array([0,0])
        self.other_pos = np.array([0,0])
        self.pos_last = np.array([0,0])
        self.euler_last = 0
        self.target = np.array([0,0])
        self.keep = np.array([0,0])
        self.count_realobs = 0
        self.wall_block = np.zeros([8,1])
        self.dist = np.array([float(0),float(0)])
        self.count = 0
        print("start wait for service")
        rospy.wait_for_service('FeatureExtract/Feature_Service')
        try:
            self.feature_client=rospy.ServiceProxy('FeatureExtract/Feature_Service', FeatureExtraction)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        print("service init success!")


    def callback(self, data):
        # print("------------------------------------3")
        gmm_result=gmm()
        subgmm_result = GMMFrame()
        subgmm_result = data.submap_gmm
        gmm_result.header=Header()
        gmm_result.header.frame_id=self.map_frame

        gmm_result.mix_num = subgmm_result.mix_num
        gmm_result.pose = data.pose_odom
        
        print("data.mix_num",gmm_result.mix_num)

        for i in range(0,gmm_result.mix_num):
            # print(gmm_result.pose.position)
            rotation_quaternion = np.asarray([gmm_result.pose.orientation.w,gmm_result.pose.orientation.x,gmm_result.pose.orientation.y,gmm_result.pose.orientation.z])
            # print("rotation_quaternion",rotation_quaternion)
            T=np.asarray([gmm_result.pose.position.x,gmm_result.pose.position.y,gmm_result.pose.position.z])
            T_qua2rota = RigidTransform(rotation_quaternion, T)
            T=T.reshape(T.shape[0],1)
            # print(T)
            R=T_qua2rota.rotation
            # print(R)
            # R_inv=np.linalg.inv(R)
            # R_T_inv=np.linalg.inv(np.transpose(R))
            # p_camera=np.asarray([data.x[i],data.y[i],data.z[i]])
            # print(np.array(data.submap_gmm.means).reshape(-1,3)[i])
            p_camera = np.array(data.submap_gmm.means).reshape(-1,3)[i]
            p_camera=p_camera.reshape(p_camera.shape[0],1)
            p_result=np.dot(R,p_camera)+T
            gmm_result.x.append(p_result[0])
            gmm_result.y.append(p_result[1])
            gmm_result.z.append(p_result[2])
            # print("------------------------------------4")

            covar=np.identity(3)
            covar[0,0]=np.array(data.submap_gmm.covariances).reshape(-1,3)[i][0]
            covar[1,1]=np.array(data.submap_gmm.covariances).reshape(-1,3)[i][1]
            covar[2,2]=np.array(data.submap_gmm.covariances).reshape(-1,3)[i][2]
            covar_result=np.dot(R,covar)
            covar_result=np.dot(covar_result,np.transpose(R))
            gmm_result.x_var.append(covar_result[0,0])
            gmm_result.y_var.append(covar_result[1,1])
            gmm_result.z_var.append(covar_result[2,2])
            # print("------------------------------------5")
        print(np.array(data.submap_gmm.weights)[i])
        gmm_result.prior=np.array(data.submap_gmm.weights)
        gmm_result.pose.position.x=0
        gmm_result.pose.position.y=0
        gmm_result.pose.position.z=0
        gmm_result.pose.orientation.x=0
        gmm_result.pose.orientation.y=0
        gmm_result.pose.orientation.z=0
        gmm_result.pose.orientation.w=1
        # print("------------------------------------6")

        self.pub.publish(gmm_result)

    def obstacle_free(self, x0, y0, x1, y1, x2, y2):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        #取机器人的位置
        # x0, y0 = eval(input("Enter coordinates for the p0 is x,y "))
        #随机点位置
        # x1, y1 = eval(input("Enter coordinates for the p1 is x,y "))
        #取每个gmm部分的（x，y）投影
        # x2, y2 = eval(input("Enter coordinates for the p2 is x,y "))
        a = y1-y0
        b = x0-x1
        c = (y0 - y1) * x0 + (x1 - x0) * y0  
        d = (a * x2 + b * y2 + c) / (np.sqrt(a**2 + b**2))
        # print("d",d) 
        if x0>x1:
            max_x=x0
            min_x=x1
        else:
            max_x=x1
            min_x=x0
        if y0>y1:
            max_y=y0 
            min_y=y1
        else:
            max_y=y1
            min_y=y0
        if abs(d) <=0.7 and min_x < x2 < max_x and min_y < y2 < max_y:
            #{x2, y2} is on the line segment from {x0, y0} to {x1, y1}"
            return 1,d
        else:
            #{x2, y2} is not on the line segment from {x0, y0} to {x1, y1}"
            return 0,d
    def target_update(self,self_id):
        exit_points = list()
        m = len(self.target_list)
        #判断原有的target_list中的点是否仍然满足条件
        while(m):
            x1 = self.target_list[m-1][0]
            y1 = self.target_list[m-1][1]
            x2 = x1 + int(self.W/2)
            y2 = y1 + int(self.L/2)
            wall = 0
            unknown = 0
            diag = 0
            # print([x2,y2])
            if 1<x2<self.W and 1<y2<self.L:
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        #判断有障碍物的点数
                        if self.gmm_map[x2 + i][y2 + j] == -1:
                            wall += 1
                        #判断有未知区域的点数
                        if self.gmm_map[x2 + i][y2 + j] == 0:
                            unknown += 1
                        if self.gmm_map[x2 + i][y2 + j] == 0 and self.gmm_map[x2 - i][y2 - j] == 1:
                            diag +=1 

            if wall > 1 or unknown <3 or diag<1 or [x1,y1] in self.gmm_wall:#不满足条件
                self.target_remove.append([x1,y1])
                self.target_list.remove([x1,y1])
            m = m -1

        #判断新的free点是否有exit_points,有的话添加上。记得self.ground在每次完成这个目标点更新程序之后要清零
        for jndex in range(len(self.gmm_ground)):   
            x1 = self.gmm_ground[jndex][0]
            y1 = self.gmm_ground[jndex][1]
            x2 = x1 - int(self.W/2)
            y2 = y1 - int(self.L/2)
            wall = 0
            unknown = 0
            diag = 0
            new = 1
            # print([x2,y2])
            if 1<x1<self.W-1 and 1<y1<self.L-3:
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        #判断有障碍物的点数
                        if self.gmm_map[x1 + i][y1 + j] == -1:
                            wall += 1
                        #判断有未知区域的点数
                        if self.gmm_map[x1 + i][y1 + j] == 0:
                            unknown += 1
                        if self.gmm_map[x1 + i][y1 + j] == 0 and self.gmm_map[x1 - i][y1 - j] == 1:
                            diag +=1 
                        if [x2 + i,y2 + j]  in self.target_list :
                            new = 0

            if wall <= 1 and unknown >=3 and diag>=1  and new ==1 and [x2,y2] not in self.target_list and [x2,y2] not in self.target_remove:
                exit_points.append([x2,y2])
        self.gmm_ground = list()

        if len(exit_points)!=0:
            num = 1
            regroup_1 =list()
            regroup_2 =list()
            regroup_3 =list()
            regroup_4 =list()
            regroup_1.append(exit_points[0])
            test = 0

            for i in range(1,len(exit_points)):
                comp = num
                # print(locals()["regroup_{}".format('3')])
                while(comp):
                    regroup_num = locals()["regroup_{}".format(comp)]
                    for j in range(len(regroup_num)):
                        if abs(exit_points[i][0] - regroup_num[j][0])<=2 and abs(exit_points[i][1] - regroup_num[j][1])<=2 :
                            test = 1
                    if test == 1:
                        regroup_num.append(exit_points[i])
                        comp = 1

                    elif test == 0 and comp == 1:
                        num = num+1
                        locals()["regroup_{}".format(num)].append(exit_points[i])
                    comp = comp -1
                    test = 0
            print(locals()["regroup_{}".format(1)])
            print(locals()["regroup_{}".format(2)])
            print(locals()["regroup_{}".format(3)])
            print(locals()["regroup_{}".format(4)])
            for n in range(1,num+1):
                self.target_list.append(self.center_point(locals()["regroup_{}".format(n)]))
                # pos = [[pos[0],pos[1]]]
                self.target_list = sorted(self.target_list, key=lambda x: (x[0]-self.pos[0])**2+(x[1]-self.pos[1])**2, reverse=False)
            print("self.target_list",self.target_list)
            # return(self.target_list)

    def center_point(self,list):
        cent_x = 0
        cent_y = 0
        center = list[0]
        for i in range(len(list)):
            cent_x = cent_x+list[i][0]
            cent_y = cent_y+list[i][1]
        cent_x = cent_x/len(list)
        cent_y = cent_y/len(list)
        distance = (list[0][0]-cent_x)**2 + (list[0][1]-cent_y)**2
        for j in range(len(list)):
            if (list[j][0]-cent_x)**2 + (list[j][1]-cent_y)**2<distance:
                center = list[j]
        return center

    def local_planner(self,self_id):
        
        if len(self.target)==0:
            self.target = self.keep
        print("self.gmm_wall",len(self.gmm_wall))

        #判断exit_point的最短路径上是否存在障碍物，若无，则直接最短路径。
        for i in range(len(self.gmm_wall)):
            # print("min(pos[0],exit_point[0])",min(pos[0],exit_point[0]))
            # print("gmm_wall[i][0]",gmm_wall[i][0])
            # print("max(pos[0],exit_point[0])",max(pos[0],exit_point[0]))
            gmm_wall_x = self.gmm_wall[i][0] -int(self.W/2)
            gmm_wall_y = self.gmm_wall[i][1] -int(self.L/2)
            #取在pos和self.target矩形框内的wall点
            if (min(self.pos[0],self.target[0])<gmm_wall_x<max(self.pos[0],self.target[0])) and (min(self.pos[1],self.target[1])<gmm_wall_y<max(self.pos[1],self.target[1])):
                #判断这些点是否会成为最短路径上的障碍
                obs_near = self.obstacle_free(self.pos[0],self.pos[1],self.target[0],self.target[1],gmm_wall_x,gmm_wall_y)[0]
                d = self.obstacle_free(self.pos[0],self.pos[1],self.target[0],self.target[1],gmm_wall_x,gmm_wall_y)[1]
                #如果是真障碍
                # print("obs_near",obs_near)
                if obs_near == 1:
                    self.count_realobs = self.count_realobs +1
            

            if (self.pos[0]-0.5 < gmm_wall_x < self.pos[0]+0.5 ) and (self.pos[1]-0.5 < gmm_wall_y < self.pos[1]+0.5):
                if gmm_wall_x>self.pos[0] and gmm_wall_y>self.pos[1]:
                    if abs(gmm_wall_x-self.pos[0])>abs(gmm_wall_y-self.pos[1]):
                        self.wall_block[4][0] +=1 
                    else:
                        self.wall_block[5][0] +=1
                elif gmm_wall_x<self.pos[0] and gmm_wall_y>self.pos[1]:
                    if abs(gmm_wall_x-self.pos[0])>abs(gmm_wall_y-self.pos[1]):
                        self.wall_block[7][0] +=1 
                    else:
                        self.wall_block[6][0] +=1
                elif gmm_wall_x>self.pos[0] and gmm_wall_y<self.pos[1]:
                    if abs(gmm_wall_x-self.pos[0])>abs(gmm_wall_y-self.pos[1]):
                        self.wall_block[3][0] +=1 
                    else:
                        self.wall_block[2][0] +=1
                elif gmm_wall_x<self.pos[0] and gmm_wall_y<self.pos[1]:
                    if abs(gmm_wall_x-self.pos[0])>abs(gmm_wall_y-self.pos[1]):
                        self.wall_block[0][0] +=1 
                    else:
                        self.wall_block[1][0] +=1
        print("self.wall_block",self.wall_block)
        print("self.count_realobs",self.count_realobs)
    def vel_ctr(self,self_id):
        #目标点距离当前位置的距离
        self.dist[0] = float(float(self.target[0])-float(self.pos[0]))
        self.dist[1] = float(float(self.target[1])-float(self.pos[1]))
        print("dist[0]",self.dist[0])
        print("dist[1]",self.dist[1])
        
        if self.dist[0]>0 and self.dist[1]>0:
            theta = math.atan(self.dist[1]/self.dist[0])
            theta = math.degrees(theta) 
            print("on the 1")

        elif self.dist[0]>0 and self.dist[1]<0:
            theta = math.atan(self.dist[1]/self.dist[0])
            theta = math.degrees(theta) 
            print("on the 3")

        elif self.dist[0]<0 and self.dist[1]>=0:
            theta = math.atan(self.dist[1]/self.dist[0])
            theta = math.degrees(theta) +180
            print("on the 2")

        elif self.dist[0]<0 and self.dist[1]<=0:
            theta = math.atan(self.dist[1]/self.dist[0])
            theta = math.degrees(theta) -180
            print("on the 4")
        
        #如果最短路径上无障碍，则直接最短路径
        if self.count_realobs<3:
            theta = theta
        #若有障碍，则进行判断
        else:
            # point1 = (pos[0],pos[1])
            # point2 = (self.target[0],self.target[1])
            # point3 = (max_point[0],max_point[1])
            # del_theta = self.cal_ang(point1,point2,point3)
            # print("del_theta",del_theta)
            # if (max_down > max_up and abs(euler)<90) or (max_down < max_up and abs(euler)>90) :
            #     theta = theta + del_theta
            #     print("+++")
            # else:
            #     theta = theta - del_theta
            #     print("---")
            ####################
            ratio = round(theta/45) + 4
            if self.wall_block[[x for x in [ratio,ratio-8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) + 0.5),(round(theta/45) + 0.5 -8)] if -4<=x<=4][0]
                theta = coef*45
                print("+ 0.5 * 45")
            elif self.wall_block[[x for x in [ratio-1,ratio-1+8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) - 0.5),(round(theta/45) - 0.5 +8)] if -4<=x<=4][0]
                theta = coef*45
                print("- 0.5 * 45")
            elif self.wall_block[[x for x in [ratio+1,ratio+1-8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) + 1.5),(round(theta/45) + 1.5 -8)] if -4<=x<=4][0]
                theta = coef*45
                print("+ 1.5 * 45")
            elif self.wall_block[[x for x in [ratio-2,ratio-2+8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) - 1.5),(round(theta/45) - 1.5 +8)] if -4<=x<=4][0]
                theta = coef*45
                print("- 1.5 * 45")
            elif self.wall_block[[x for x in [ratio+2,ratio+2-8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) + 2.5),(round(theta/45) + 2.5 -8)] if -4<=x<=4][0]
                theta = coef*45
                print("+ 2.5 * 45")
            elif self.wall_block[[x for x in [ratio-3,ratio-3+8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) - 2.5),(round(theta/45) - 2.5 +8)] if -4<=x<=4][0]
                theta = coef*45
                print("- 2.5 * 45")
            elif self.wall_block[[x for x in [ratio+3,ratio+3-8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) + 3.5),(round(theta/45) + 3.5 -8)] if -4<=x<=4][0]
                theta = coef*45
                print("+ 3.5 * 45")
            elif self.wall_block[[x for x in [ratio-4,ratio-4+8] if 0<=x<=7][0]][0] < 2:
                coef = [x for x in [(round(theta/45) - 3.5),(round(theta/45) - 3.5 +8)] if -4<=x<=4][0]
                theta = coef*45
                print("- 3.5 * 45")
            else:
                coef = [x for x in [(round(theta/45) + 4),(round(theta/45) + 4 -8)] if 0<=x<=8][0]
                theta = coef*45
                print("+ 4 * 45")
              
        print("theta",theta)
        print("euler",self.euler)

        if abs(self.euler_last - self.euler)<3 and self.cu_mix_num>100 and abs(self.pos_last[0] - self.pos[0])<0.05 and abs(self.pos_last[1] - self.pos[1])<0.05:
            target_linear_vel = -0.03
            target_angular_vel = 0
            print(self.vels(target_linear_vel,target_angular_vel))
        elif abs(theta - self.euler) <=15 or abs(theta - self.euler) >=330 :
            target_linear_vel = 0.35
            target_angular_vel = 0
            print(self.vels(target_linear_vel,target_angular_vel))
        elif 0<=theta - self.euler<=180 or self.euler - theta>=180:
            target_linear_vel = 0.0
            target_angular_vel = 0.2
            print(self.vels(target_linear_vel,target_angular_vel))
        else:
            target_linear_vel = 0.0
            target_angular_vel = -0.2
            print(self.vels(target_linear_vel,target_angular_vel))
        
        if  self.cu_mix_num<460:
            target_linear_vel = 0
            target_angular_vel = 0.1888
            print(self.vels(target_linear_vel,target_angular_vel))

        gmm_exit = 0
        self.keep = self.target
        self.count_realobs = 0  
        print("self.keep",self.keep)
        self.euler_last = self.euler
        self.pos_last = self.pos
        #发布cmd_vel指令
        twist = Twist()           
        twist.linear.x = target_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = target_angular_vel           
        self.ctr_pub.publish(twist)

    def vels(self,target_linear_vel, target_angular_vel):
        return "currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel)

    def mr_target(self,data): #维护另一个机器人的位置信息
        robot_id = data.data[2]
        robot_pos = np.array([data.data[0],data.data[1]])
        robot_target = np.array([data.data[3],data.data[4]])
        if robot_id != self.self_robot_id:
            # print("robot_id",robot_id)
            self.other_pos = robot_pos
            self.other_target = robot_target
            print("self.other_pos",self.other_pos)

    def other_location(self,data):
        self.co_explore = data.data[0]

    def callback_nav(self, data): #监听pointcloud,自己新建地图,并且保存对应的 odom.
        print(datetime.now()) 
        self.count = self.count +1
        assert isinstance(data, gmm)
        pointheader = data.header
        pointtime = pointheader.stamp
        try:
            # print("self.other_pos",self.other_pos)
            self.tf_listener.waitForTransform(self.map_frame, self.baselink_frame, pointtime, rospy.Duration(5.0))    
            #等待Duration=5s，判断map是否转换为base_link
            #得到机器人当前位置pos 和 姿态eular
            (self.pos,self.ori) = self.tf_listener.lookupTransform(self.map_frame, self.baselink_frame,pointtime)
            self.pos = np.array([self.pos[0],self.pos[1]])
            print("self.pos",self.pos)
            # print("self.ori",self.ori)
            self.euler = transformations.euler_from_quaternion(self.ori)    #将四元数转换为欧拉角
            self.euler = math.degrees(self.euler[2])
            cu_gmm = data
            self.cu_mix_num = cu_gmm.mix_num
            
            print("cu_mix_num",self.cu_mix_num)

            if self.co_explore == 1:
                current_pos = np.array([self.pos[0],self.pos[1],self.self_robot_id,self.target[0],self.target[1]])
                current_pos = Float32MultiArray(data = current_pos)
                self.pub_location.publish(current_pos)
                rospy.loginfo(current_pos.data)
                        
            for index in range(self.cu_mix_num):#最新的submap
                x = cu_gmm.x[index]         
                y = cu_gmm.y[index]               
                z = cu_gmm.z[index]
                if -7.5<x<7.5 and -11.5<y<11.5:
                    x1 = round(x + int(self.W/2))
                    y1 = round(y + int(self.L/2))

                    #记录墙上的点
                    if z>=0.7:
                        if not [x1,y1] in self.gmm_wall_up:
                            self.gmm_wall_up.append([x1,y1])
                    elif 0.7>z>0.1:
                        if not [x1,y1] in self.gmm_wall_down:
                            self.gmm_wall_down.append([x1,y1])
                    #记录地面的点
                    elif z<=0.1:
                        #考虑需不需要拓展8领域
                        # for i in [-1,0,1]:
                        #     for j in [-1,0,1]:
                        self.gmm_map[x1][y1] = 1
                        if self.cu_mix_num>460:
                            if not [x1,y1] in self.gmm_ground :
                                self.gmm_ground.append([x1,y1])
                        else:
                            self.gmm_ground.append([x1,y1])

                        if not [x1,y1] in self.gmm_ground_all :
                            self.gmm_ground_all.append([x1,y1])
                    
            print("gmm_ground",len(self.gmm_ground))
            for i in range(0,17):
                for j in range(0,25):
                    if [i,j] in self.gmm_wall_down and [i,j] in self.gmm_wall_up and [i,j] not in self.gmm_wall:
                        self.gmm_wall.append([i,j])
                        self.gmm_map[i][j] = -1
            # print("self.gmm_wall",self.gmm_wall)

            #去除ground中的obstacle
            u = len(self.gmm_ground)
            while(u):            
                if self.gmm_ground[u-1] in self.gmm_wall:
                    self.gmm_ground.remove(self.gmm_ground[u-1])
                u = u -1

            #如果在target中有与现在位置靠近的，删去。
            n = len(self.target_list)
            while(n):
                if (abs(self.pos[0]-self.target_list[n-1][0])<1 and abs(self.pos[1]-self.target_list[n-1][1])<1):
                    self.target_remove.append(self.target_list[n-1])
                    self.target_list.remove(self.target_list[n-1])               
                    print("remove near!!")
                n = n -1

            #不靠近预设的目标点附近，不能换新的目标点
            if (abs(self.pos[0]-self.keep[0])<1 and abs(self.pos[1]-self.keep[1])<1) or (self.target[0] == 0 and self.target[1] == 0) or self.cu_mix_num<500:
                if [self.keep[0],self.keep[1]] in self.target_list and (abs(self.pos[0]-self.keep[0])<1 and abs(self.pos[1]-self.keep[1])<1):
                    print("remove!!")
                    self.target_list.remove(self.keep)
                    self.target_remove.append(self.keep)
                #到达最近目标点后，从point序列中去除

                self.target_update(self)

                if [self.keep[0],self.keep[1]] in self.target_list and (abs(self.pos[0]-self.keep[0])<1 and abs(self.pos[1]-self.keep[1])<1):
                    print("remove!!")
                    self.target_list.remove(self.keep)
                    self.target_remove.append(self.keep)

                # #对target list的每一项计算score
                # if len(self.target_list)!=0 :
                #     if self.other_pos[0]!=0 and self.other_pos[1]!=0:
                #     # self.target = self.target_list[0]
                #         score = np.zeros((len(self.target_list),1))
                #         score[0] = (math.hypot(self.target_list[0][0]-self.other_pos[0],self.target_list[0][1]-self.other_pos[1])/math.hypot(self.target_list[0][0]-self.pos[0],self.target_list[0][1]-self.pos[1]))
                #         Max = score[0]
                #         print("score[0]",score[0])
                #         max_num = 0
                #         for z in range(len(self.target_list)):
                #             score[z] = (math.hypot(self.target_list[z][0]-self.other_pos[0],self.target_list[z][1]-self.other_pos[1])/math.hypot(self.target_list[z][0]-self.pos[0],self.target_list[z][1]-self.pos[1]))
                #             if abs(self.target_list[z][0]-self.other_target[0])<=1 and abs(self.target_list[z][1]-self.other_target[1])<=1:
                #                 score[z] = 0
                #             if score[z] > Max:
                #                 Max = score[z]
                #                 max_num = z
                #         print("computation over!!!!!!!!!!!!!!!!!!!")
                #         print(score)
                #         self.target = self.target_list[max_num]
                #     else:
                self.target = self.target_list[0]

            else:
                self.target = self.keep
            self.keep = self.target
            

            # #发布coverage-time
            # ratio_of_coverage = float(len(self.gmm_ground_all)/(self.W*self.L-16))
            # t = float((datetime.now()-self.begin).seconds)
            # arr = np.array([ratio_of_coverage,t])
            # arr = Float32MultiArray(data = arr)
            # self.cov_time.publish(arr)

            # #发布path_length-time
            # if self.pos_last[0]!=0 and self.pos_last[1]!=0:
            #     step_length = float(math.sqrt((self.pos[0]-self.pos_last[0])**2 + (self.pos[1]-self.pos_last[1])**2))
            #     self.step_sum = float(self.step_sum + step_length)
            #     t = float((datetime.now()-self.begin).seconds)
            #     arr_length = np.array([self.step_sum,ratio_of_coverage])
            #     arr_length = Float32MultiArray(data = arr_length)
            #     self.length_time.publish(arr_length)
            #     print("sum_length",self.step_sum)

            print("self.target_list",self.target_list)
            print("self.target",self.target)

            #发布当前position以及orientation
            # current_pos = np.array([self.pos[0],self.pos[1],self.ori[0],self.ori[1],self.ori[2],self.ori[3]])
            # current_pos = Float32MultiArray(data = current_pos )
            # self.pub_pos.publish(current_pos)
            # rospy.loginfo(current_pos.data)

            # if (self.count-3)%5 == 0:
            #     print("publish target!!!")
            #     target_pos = np.array([self.target[0],self.target[1]])
            #     target_pos = Float32MultiArray(data = target_pos )
            #     self.pub_target.publish(target_pos)


            if  self.cu_mix_num<460:
                print(self.vels(0,0.1888))
                #发布cmd_vel指令
                twist = Twist()           
                twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
                twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.1888           
                self.ctr_pub.publish(twist)
            else:
                target_pos = np.array([self.target[0],self.target[1]])
                target_pos = Float32MultiArray(data = target_pos )
                self.pub_target.publish(target_pos)



                
            # self.local_planner(self)
            # self.vel_ctr(self)
            
        except TransformException:
            rospy.loginfo("tf Error")
            rospy.Duration(1.0)

    def cal_theta(self,initial,destination):
        dist = np.array([0.0,0.0])
        dist[1] = float(float(destination[1])-float(initial[1]))
        dist[0] = float(float(destination[0])-float(initial[0]))
        # print("dist[1]",dist[1])
        # print("dist[0]",dist[0])
        angle = 0
        if dist[0]>0 and dist[1]>0: #用pi表示
            angle = math.atan(dist[1]/dist[0])
            print("on the 1")

        elif dist[0]>0 and dist[1]<0:
            angle = math.atan(dist[1]/dist[0])
            print("on the 4")

        elif dist[0]<0 and dist[1]>=0:
            angle = math.atan(dist[1]/dist[0]) + math.pi
            print("on the 2")

        elif dist[0]<0 and dist[1]<=0:
            angle = math.atan(dist[1]/dist[0]) - math.pi
            print("on the 3")
        # print("angle",angle)
        return angle

    def quat_to_euler(self,x,y,z,w):
        r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        p = math.asin(2*(w*y-z*x))
        y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        angleR = r*180/math.pi #Y
        angleP = p*180/math.pi #X
        angleY = y*180/math.pi #Z
        return angleP,angleR,angleY

    def prob_visual(self, gmm_map):
        global resulotion, z_const
        Xbottom=gmm_map.x[gmm_map.x.index(min(gmm_map.x))]-2  # is gmm_map.x a list?
        Xtop=gmm_map.x[gmm_map.x.index(max(gmm_map.x))]+2
        Ybottom=gmm_map.y[gmm_map.y.index(min(gmm_map.y))]-2
        Ytop=gmm_map.y[gmm_map.y.index(max(gmm_map.y))]+2
        pro=np.zeros((int((Xtop-Xbottom)/resolution+1),int((Ytop-Ybottom)/resolution+1)))
        print("pro.shape",pro.shape)
        print("Xbottom,Xtop,Ybottom,Ytop",Xbottom,Xtop,Ybottom,Ytop)
        num_tmp=0
        for i in range(0,gmm_map.mix_num):
            if (gmm_map.z[i]>0.1):
                num_tmp=num_tmp+1
        mu=np.zeros((num_tmp,3))
        sigma=np.zeros((num_tmp,3,3))
        prior=np.zeros((num_tmp,1))
        ptr=0
        for p in range(0,gmm_map.mix_num):
            if (gmm_map.z[p]>0.1):
                mu[ptr,:]=[gmm_map.x[p],gmm_map.y[p],gmm_map.z[p]]
                sigma[ptr,:,:]=np.diag([gmm_map.x_var[p],gmm_map.y_var[p],gmm_map.z_var[p]])
                prior[ptr]=gmm_map.prior[p]
                ptr=ptr+1
        ptr=0
        # print("mu:",mu)
        # print("sigma:",sigma)
        # print("prior:",prior)
        for x in np.arange(Xbottom,Xtop,resolution):
            y= np.arange(Ybottom,Ytop,resolution)
            y=y.reshape(len(y),1)    
            L=int((Ytop-Ybottom)/resolution+1)
            x_temp=np.ones([L,1])*x
            z_temp=np.ones([L,1])*z_const
            pt_temp=np.hstack((x_temp,y,z_temp))
            totalpdf=np.zeros((L,1))
            for pp in range(0,num_tmp):
                Npdf=st.multivariate_normal.pdf(pt_temp,mu[pp,:],sigma[pp,:,:])
                totalpdf=totalpdf+Npdf.reshape(len(Npdf),1)*prior[pp]
            pro[ptr,:]=totalpdf.reshape(pro[ptr,:].shape)
            ptr=ptr+1
        # print(pro)
        return pro,Xtop,Xbottom,Ytop,Ybottom

    def subgmm_cb(self,data):
        global gmm_cnt
        global gmm_map
        gmm_cnt=gmm_cnt+1
        print("gmm_cnt",gmm_cnt)
        gmm_map=data
        

    def gmm_nav(self,self_id):
        print("has already navigation!!!!!!!!!!!!!!!!!!!!")
        start=time.time()
        global z_const, gmm_map, resolution, path_pub, tra_total, ang_total
        [pro,Xtop,Xbottom,Ytop,Ybottom]=self.prob_visual(gmm_map)
        # begin=[int((begin_pos[0]-Xbottom)/resolution),int((begin_pos[1]-Ybottom)/resolution)]
        radius=1 #dong
        print("radius",radius)
        radius=max(math.sqrt((self.pos[0]-self.target[0])**2 + (self.pos[1]-self.target[1])**2)/3, 1) #步长随离target的距离调整，但不能小于1
        [Fx,Fy]=np.gradient(pro)
        print("Fx: ", Fx.shape)
        print("Fy: ", Fy.shape)
        # print("begin_pos",begin_pos)
        print("target",self.target)
        print("self.pos",self.pos)
        # dong的表示，但是只有-90～90的角度变化
        # angle_ulti=math.atan((self.target[1]-self.pos[1])/(self.target[0]-self.pos[0])) #angle_ulti为target->pos

        # 分象限
        angle_ulti = self.cal_theta(self.pos,self.target) #pi形式
        print("angle_ulti",angle_ulti)
        goal=[self.pos[0]+radius*math.cos(angle_ulti),self.pos[1]+radius*math.sin(angle_ulti)] # position in the real world
        print("goal",goal)
        F1_tmp=Fx[int((goal[0]-Xbottom)/resolution),int((goal[1]-Ybottom)/resolution)]
        F2_tmp=Fy[int((goal[0]-Xbottom)/resolution),int((goal[1]-Ybottom)/resolution)]
        # angle_x=math.atan((goal[1]-self.pos[1])/(goal[0]-self.pos[0]))/math.pi*180  #angle_x为goal->pos
        angle_x=self.cal_theta(self.pos, goal) /math.pi*180
        print("angle_x",angle_x)
        F_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
        print("F_now: ", F_now)
        if (F_now>(1e-20)):#1e-25
            for p in range(1,4):
                angle1=angle_x+p*30
                # angle1=angle_x * (-1)**(int((angle_x+p*30)/180))+p*30
                angle1=angle1*math.pi/180
                goal1_tmp=[self.pos[0]+radius*math.cos(angle1),self.pos[1]+radius*math.sin(angle1)]
                F1_tmp=Fx[int((goal1_tmp[0]-Xbottom)/resolution),int((goal1_tmp[1]-Ybottom)/resolution)]
                F2_tmp=Fy[int((goal1_tmp[0]-Xbottom)/resolution),int((goal1_tmp[1]-Ybottom)/resolution)]
                F1_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
                angle2=angle_x-p*30
                # angle2=angle_x * (-1)**(int((angle_x-p*30)/180))-p*30
                angle2=angle2*math.pi/180
                goal2_tmp=[self.pos[0]+radius*math.cos(angle2),self.pos[1]+radius*math.sin(angle2)]
                F1_tmp=Fx[int((goal2_tmp[0]-Xbottom)/resolution),int((goal2_tmp[1]-Ybottom)/resolution)]
                F2_tmp=Fy[int((goal2_tmp[0]-Xbottom)/resolution),int((goal2_tmp[1]-Ybottom)/resolution)]
                F2_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
                # print("angle1",angle1)
                # print("angle2",angle2)

                if (min(F1_now,F2_now)<F_now):#计算目标角度angle_x
                    if (F1_now<F2_now):
                        goal=goal1_tmp
                        # angle_x=math.atan((goal[1]-self.pos[1])/(goal[0]-self.pos[0]))/math.pi*180
                        angle_x=self.cal_theta(self.pos,goal) /math.pi*180
                        # print("angle_x",angle_x)
                    else :
                        goal=goal2_tmp
                        # angle_x=math.atan((goal[1]-self.pos[1])/(goal[0]-self.pos[0]))/math.pi*180
                        angle_x=self.cal_theta(self.pos,goal) /math.pi*180
                        # print("angle_x",angle_x)
                    break 
        
        tra_total=tra_total+radius
        # [angleX,angleY,angleZ]=self.quat_to_euler(self.ori[0],self.ori[1],self.ori[2],self.ori[3])#dong:得到当前角度angleZ
        angleZ = transformations.euler_from_quaternion(self.ori)    #ours :得到当前角度angleZ
        angleZ = math.degrees(angleZ[2]) #ours

        ang_total=ang_total+ abs(angle_x-angleZ)       
        print("tra_total= ", tra_total)
        print("ang_total= ", ang_total)
        end=time.time()
        # print("----------------------------------navigation_time: ", end-start)

        ##vel pub
        stop = Twist()
        self.ctr_pub.publish(stop)
        # [angleX,angleY,angleZ]=self.quat_to_euler(self.ori[0],self.ori[1],self.ori[2],self.ori[3])
        angleZ = transformations.euler_from_quaternion(self.ori)    #ours
        angleZ = math.degrees(angleZ[2]) #ours


        #rotate
        while(abs(angleZ-angle_x)>4):#0.02
            (self.pos,self.ori) = self.tf_listener.lookupTransform(self.map_frame, self.baselink_frame,rospy.Time(0)) #实时更新位置和姿态
            self.pos = np.array([self.pos[0],self.pos[1]])
            move=Twist()
            if 180>(angle_x-angleZ)>0 or (angle_x-angleZ)<-180:
                move.angular.z=0.25 #0.1
            else:
                move.angular.z=-0.25 #0.1

            self.ctr_pub.publish(move)      
            rospy.sleep(0.1)       
            self.ctr_pub.publish(stop)
            rospy.sleep(0.1) 

            # [angleX,angleY,angleZ]=self.quat_to_euler(self.ori[0],self.ori[1],self.ori[2],self.ori[3])
            angleZ = transformations.euler_from_quaternion(self.ori)    #ours
            angleZ = math.degrees(angleZ[2]) #ours
            # print("angle:")
            # print(angleZ)
            # print(angle_x)

        self.ctr_pub.publish(stop)
        rospy.sleep(1) 
        # print(self.pos)
        # print(goal)

        #translate
        end=time.time()
        self.pos_last =self.pos
        while(abs(self.pos[0]-goal[0])>0.1 or abs(self.pos[1]-goal[1])>0.1):
            (self.pos,self.ori) = self.tf_listener.lookupTransform(self.map_frame, self.baselink_frame,rospy.Time(0)) #实时更新位置和姿态
            self.pos = np.array([self.pos[0],self.pos[1]])
            # print("position:")
            # print(self.pos)
            # print(goal)
            move=Twist()
            vel_const=max(radius/5, 0.15) #线速度随步长变化,但不能小于0.15
            # move.linear.x=math.cos(angle_x*math.pi/180)*vel_const*np.sign(goal[0]-self.pos[0])  #dong:差速控制
            # move.linear.y=math.sin(angle_x*math.pi/180)*vel_const*np.sign(goal[1]-self.pos[1])
            move.linear.x=vel_const
            if math.cos(angle_x*math.pi/180)*np.sign(goal[0]-self.pos[0])<0 or math.sin(angle_x*math.pi/180)*np.sign(goal[1]-self.pos[1])<0:#当超过时，break
                break
            self.ctr_pub.publish(move)
            rospy.sleep(0.1)       
            # self.ctr_pub.publish(stop)
        self.ctr_pub.publish(stop)
        # print("-------------------------------control time: ", time.time()-end)

        #发布coverage-time
        ratio_of_coverage = float(len(self.gmm_ground_all)/(self.W*self.L-16))
        self.t = self.t + time.time()-end
        arr = np.array([ratio_of_coverage,self.t])
        arr = Float32MultiArray(data = arr)
        self.cov_time.publish(arr)

        #发布path_length-coverage
        step_length = float(math.sqrt((self.pos[0]-self.pos_last[0])**2 + (self.pos[1]-self.pos_last[1])**2))
        self.step_sum = float(self.step_sum + step_length)
        arr_length = np.array([self.step_sum,ratio_of_coverage])
        arr_length = Float32MultiArray(data = arr_length)
        self.length_time.publish(arr_length)
        print("sum_length",self.step_sum)

        if (abs(self.pos[0]-self.target[0])+abs(self.pos[1]-self.target[1])<radius):
            print("tra_total= ", tra_total)
            print("ang_total= ", ang_total)    
            # while (1):
            print("Target Reached! Navigation Stop!")
            rospy.sleep(0.5)

                




def main():
    global gmm_cnt
    global goal_cnt
    global gmm_map
    global resolution, z_const
    global tra_total, ang_total
    tra_total=0
    ang_total=0
    resolution=0.25
    z_const=0.5
    gmm_cnt=0
    goal_cnt=0
    rospy.init_node('nav', anonymous=True)
    Robot1 = Navigation(rospy.get_param('~robot_id',2))

    rospy.spin()

if __name__ == "__main__":
    main()       