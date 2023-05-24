#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import numpy as np
import math
import xlwt
import scipy.stats as st
from datetime import datetime
from autolab_core import RigidTransform

from tf_conversions import transformations
import tf
import tf2_ros
from tf2_ros import TransformException

from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import TransformStamped, Twist
from gmm_map_python.msg import gmm, Submap
from gmm_map_python.srv import *

from GMMmap import GMMFrame

LIN_VEL_STEP_SIZE = 0.24
ANG_VEL_STEP_SIZE = 0.18

class Navigation:
    def __init__(self, self_id):
        self.self_robot_id = self_id
        self.tf_listener = tf.TransformListener()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.transform_base_camera = TransformStamped()
        self.baselink_frame = rospy.get_param('~baselink_frame','base_link')
        self.odom_frame = rospy.get_param('~odom_frame','odom')
        self.map_frame = rospy.get_param('~map_frame','map')
        self.ctr_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_subgmm = rospy.Subscriber('GMMglobal',gmm, self.callback_nav,queue_size=10)
        self.cov_time = rospy.Publisher('coverage',Float32MultiArray,queue_size=10)
        self.length_time = rospy.Publisher('length',Float32MultiArray,queue_size=10)
        self.W = 17
        self.L = 25
        self.step_sum = 0
        self.begin = datetime.now()
        self.gmm_map = np.zeros((self.W,self.L))
        self.target_list = list()
        self.target_remove = list()
        self.gmm_ground = list()
        self.gmm_ground_new = list()
        self.gmm_wall_up = list()
        self.gmm_wall_down = list()
        self.gmm_wall = list()
        self.gmm_ground_all = list()
        self.pos = 0
        self.ori = 0
        self.euler = 0
        self.pos_last = 0
        self.euler_last = 0
        self.target = np.array([0,0])
        self.keep = np.array([0,0])
        self.count_realobs = 0
        self.wall_block = np.zeros([8,1])
        self.dist = np.array([float(0),float(0)])
        print("start wait for service")
        rospy.wait_for_service('FeatureExtract/Feature_Service')
        try:
            self.feature_client=rospy.ServiceProxy('FeatureExtract/Feature_Service', FeatureExtraction)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        print("service init success!")

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
        if abs(d) <=1 and min_x < x2 < max_x and min_y < y2 < max_y:
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
            if 1<x1<self.W-1 and 1<y1<self.L-1:
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
        if self.count_realobs<1:
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

        if abs(self.euler_last - self.euler)<3 and self.cu_mix_num>100 and abs(self.pos_last[0] - self.pos[0])<0.1 and abs(self.pos_last[1] - self.pos[1])<0.1:
            target_linear_vel = -0.1
            target_angular_vel = 0
            print(self.vels(target_linear_vel,target_angular_vel))
        elif abs(theta - self.euler) <=15 or abs(theta - self.euler) >=330 :
            target_linear_vel = 0.24
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

    def callback_nav(self, data): #监听pointcloud,自己新建地图,并且保存对应的 odom.
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
        assert isinstance(data, gmm)
        pointheader = data.header
        pointtime = pointheader.stamp
        try:
            self.tf_listener.waitForTransform(self.map_frame, self.baselink_frame, pointtime, rospy.Duration(5.0))    
            #等待Duration=5s，判断map是否转换为base_link
            #得到机器人当前位置pos 和 姿态eular
            (self.pos,self.ori) = self.tf_listener.lookupTransform(self.map_frame, self.baselink_frame,pointtime)
            print("self.pos",self.pos)
            self.euler = transformations.euler_from_quaternion(self.ori)    #将四元数转换为欧拉角
            self.euler = math.degrees(self.euler[2])
            cu_gmm = data
            self.cu_mix_num = cu_gmm.mix_num
            
            print("cu_mix_num",self.cu_mix_num)
                        
            for index in range(self.cu_mix_num):#最新的submap
                x = cu_gmm.x[index]         
                y = cu_gmm.y[index]               
                z = cu_gmm.z[index]
                x1 = round(x + int(self.W/2))
                y1 = round(y + int(self.L/2))

                #记录墙上的点
                if z>=0.1:
                    if not [x1,y1] in self.gmm_wall:
                        self.gmm_wall.append([x1,y1])
                        self.gmm_map[x1][y1] = -1

                #记录地面的点
                elif z<0.1:
                    #考虑需不需要拓展8领域
                    # for i in [-1,0,1]:
                    #     for j in [-1,0,1]:
                    self.gmm_map[x1][y1] = 1
                    if self.cu_mix_num>460:
                        if not [x1,y1] in self.gmm_ground :
                            self.gmm_ground.append([x1,y1])
                            self.gmm_ground_new.append([x1,y1])
                    else:
                        self.gmm_ground.append([x1,y1])
                        self.gmm_ground_new.append([x1,y1])

                    if not [x1,y1] in self.gmm_ground_all :
                        self.gmm_ground_all.append([x1,y1])
                    
            print("gmm_ground",len(self.gmm_ground))



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
                #到达最近目标点后，从point序列中去除
                self.target_update(self)

                if [self.keep[0],self.keep[1]] in self.target_list and (abs(self.pos[0]-self.keep[0])<1 and abs(self.pos[1]-self.keep[1])<1):
                    print("remove!!")
                    self.target_list.remove(self.keep)
                if len(self.target_list)!=0:
                    self.target = self.target_list[0]
                
            else:
                self.target = self.keep

            ratio_of_coverage = float(len(self.gmm_ground_all)/(self.W*self.L-16))
            t = float((datetime.now()-self.begin).seconds)
            arr = np.array([ratio_of_coverage,t])
            arr = Float32MultiArray(data = arr)
            rospy.loginfo(arr.data)
            print("ratio of coverage",ratio_of_coverage)
            print("TIME:",t)
            
            if self.pos_last!=0:
                step_length = float(math.sqrt((self.pos[0]-self.pos_last[0])**2 + (self.pos[1]-self.pos_last[1])**2))
                self.step_sum = float(self.step_sum + step_length)
                t = float((datetime.now()-self.begin).seconds)
                arr_length = np.array([self.step_sum,ratio_of_coverage])
                arr_length = Float32MultiArray(data = arr_length)
                self.length_time.publish(arr_length)
                print("sun_length",self.step_sum)

            # self.excelsheet.write(self.count,0,(datetime.now()-self.begin).seconds)
            # self.excelsheet.write(self.count,1,len(self.gmm_ground_all)/(self.W*self.L-16))
            # self.coverage_time.save('cov_time.txt')
            self.cov_time.publish(arr)
            

            print("self.target_list",self.target_list)
            print("self.target",self.target)

            self.local_planner(self)
            self.vel_ctr(self)

            


        except TransformException:
            rospy.loginfo("tf Error")
            rospy.Duration(1.0)

def main():

    rospy.init_node('nav', anonymous=True)
    Robot1 = Navigation(rospy.get_param('~robot_id',1))
    rospy.spin()

if __name__ == "__main__":
    main()       