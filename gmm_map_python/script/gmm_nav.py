#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import signal
import sys
import rospy
import copy
import numpy as np
import scipy.stats as st
import math
import time
from std_msgs.msg import Header, String,Float32MultiArray,Float32
from geometry_msgs.msg import Point,Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped, Twist
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path
from gmm_map_python.msg import gmm, gmmlist

# gmm_cnt=0
# goal_cnt=0

def subgmm_cb(data):
    global gmm_cnt
    global gmm_map
    gmm_cnt=gmm_cnt+1
    gmm_map=data
    # if (gmm_cnt!=0):
    #     if(data.prior[1]!=prev_data.prior[1]):
    #         gmm_cnt=gmm_cnt+1
    #         gmm_map=data
    # else:
    #     gmm_map=data
    #     gmm_cnt=gmm_cnt+1

    # target is the ultimate navigation point, goal is the next navigation step    
def goal_cb(data):
    global goal_cnt, target
    target=[data.pose.position.x,data.pose.position.y]
    goal_cnt=goal_cnt+1

def begin_cb(data):
    global begin_pos, begin_quat
    # pose[1] is the second element, with the name "turtlebot3"
    begin_pos=[data.data[0],data.data[1]]
    print("begin_pos",begin_pos[0], begin_pos[1])
    begin_quat=[data.pose[1].orientation.x,data.pose[1].orientation.y,data.pose[1].orientation.z,data.pose[1].orientation.w]

def prob_visual(gmm_map):
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

def gmm_nav():
    start=time.time()
    global z_const, gmm_map, resolution, begin_pos, begin_quat, target, move_pub, path_pub, tra_total, ang_total
    [pro,Xtop,Xbottom,Ytop,Ybottom]=prob_visual(gmm_map)
    # begin=[int((begin_pos[0]-Xbottom)/resolution),int((begin_pos[1]-Ybottom)/resolution)]
    radius=0.3
    [Fx,Fy]=np.gradient(pro)
    print("Fx: ", Fx.shape)
    print("Fy: ", Fy.shape)
    # print("begin_pos",begin_pos)
    print("target",target)
    angle_ulti=math.atan((target[1]-begin_pos[1])/(target[0]-begin_pos[0]))
    goal=[begin_pos[0]+radius*math.cos(angle_ulti),begin_pos[1]+radius*math.sin(angle_ulti)] # position in the real world
    print("gaol",goal)
    F1_tmp=Fx[int((goal[0]-Xbottom)/resolution),int((goal[1]-Ybottom)/resolution)]
    F2_tmp=Fy[int((goal[0]-Xbottom)/resolution),int((goal[1]-Ybottom)/resolution)]
    angle_x=math.atan((goal[1]-begin_pos[1])/(goal[0]-begin_pos[0]))/math.pi*180

    F_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
    print("F_now: ", F_now)
    if (F_now>(1e-25)):
        for p in range(1,4):
            angle1=angle_x+p*30
            angle1=angle1*math.pi/180
            goal1_tmp=[begin_pos[0]+radius*math.cos(angle1),begin_pos[1]+radius*math.sin(angle1)]
            F1_tmp=Fx[int((goal1_tmp[0]-Xbottom)/resolution),int((goal1_tmp[1]-Ybottom)/resolution)]
            F2_tmp=Fy[int((goal1_tmp[0]-Xbottom)/resolution),int((goal1_tmp[1]-Ybottom)/resolution)]
            F1_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
            angle2=angle_x-p*30
            angle2=angle2*math.pi/180
            goal2_tmp=[begin_pos[0]+radius*math.cos(angle2),begin_pos[1]+radius*math.sin(angle2)]
            F1_tmp=Fx[int((goal2_tmp[0]-Xbottom)/resolution),int((goal2_tmp[1]-Ybottom)/resolution)]
            F2_tmp=Fy[int((goal2_tmp[0]-Xbottom)/resolution),int((goal2_tmp[1]-Ybottom)/resolution)]
            F2_now=np.sqrt(pow(F1_tmp,2)+pow(F2_tmp,2))
            if (min(F1_now,F2_now)<F_now):
                if (F1_now<F2_now):
                    goal=goal1_tmp
                    angle_x=math.atan((goal[1]-begin_pos[1])/(goal[0]-begin_pos[0]))/math.pi*180
                else :
                    goal=goal2_tmp
                    angle_x=math.atan((goal[1]-begin_pos[1])/(goal[0]-begin_pos[0]))/math.pi*180
                break 
        
    tra_total=tra_total+radius
    [angleX,angleY,angleZ]=quat_to_euler(begin_quat[0],begin_quat[1],begin_quat[2],begin_quat[3])
    ang_total=ang_total+ abs(angle_x-angleZ)       
    print("tra_total= ", tra_total)
    print("ang_total= ", ang_total)
    end=time.time()
    print("----------------------------------navigation_time: ", end-start)

    ##vel pub
    stop = Twist()
    move_pub.publish(stop)
    [angleX,angleY,angleZ]=quat_to_euler(begin_quat[0],begin_quat[1],begin_quat[2],begin_quat[3])

    #rotate
    while(abs(angleZ-angle_x)>0.02):
        move=Twist()
        move.angular.z=np.sign(angle_x-angleZ)*0.1
        move_pub.publish(move)      
        rospy.sleep(0.1)       
        move_pub.publish(stop)
        rospy.sleep(0.1) 
        [angleX,angleY,angleZ]=quat_to_euler(begin_quat[0],begin_quat[1],begin_quat[2],begin_quat[3])
        print("angle:")
        print(angleZ)
        print(angle_x)
    move_pub.publish(stop)
    rospy.sleep(1) 
    # print(begin_pos)
    # print(goal)

    #translate
    while(abs(begin_pos[0]-goal[0])>0.03 or abs(begin_pos[1]-goal[1])>0.03):
        print("position:")
        print(begin_pos)
        print(goal)
        move=Twist()
        vel_const=0.1
        move.linear.x=math.cos(angle_x*math.pi/180)*vel_const*np.sign(goal[0]-begin_pos[0])
        move.linear.y=math.sin(angle_x*math.pi/180)*vel_const*np.sign(goal[1]-begin_pos[1])
        move_pub.publish(move)
        rospy.sleep(0.1)       
        move_pub.publish(stop)
    move_pub.publish(stop)
    print("-------------------------------control time: ", time.time()-end)
    if (abs(begin_pos[0]-target[0])+abs(begin_pos[1]-target[1])<radius):
        
        print("tra_total= ", tra_total)
        print("ang_total= ", ang_total)    
        while (1):
            print("Target Reached! Navigation Stop!")
            rospy.sleep(0.5)

    #angle jiaosudu === linar xiansudu 

def quat_to_euler(x,y,z,w):
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

    angleR = r*180/math.pi #Y
    angleP = p*180/math.pi #X
    angleY = y*180/math.pi #Z
    return angleP,angleR,angleY

def main():
    global gmm_cnt
    global goal_cnt
    global gmm_map
    global target
    global begin_pos, begin_quat
    global resolution, z_const
    global move_pub, goal_pub
    global tra_total, ang_total
    tra_total=0
    ang_total=0
    resolution=0.5
    z_const=0.5
    gmm_cnt=0
    goal_cnt=0
    rospy.init_node('gmm_nav', anonymous=True)
    rate = rospy.Rate(5)
    move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    goal_pub = rospy.Publisher('/goal_path', Path, queue_size=1)
    target_sub = rospy.Subscriber('/move_base_simple/goal',PoseStamped, goal_cb,queue_size=100)
    gmm_sub = rospy.Subscriber('gmm_after_trans',gmm, subgmm_cb,queue_size=100)
    begin_pose_sub = rospy.Subscriber('/robot1/target_pos',Float32MultiArray, begin_cb,queue_size=100)
    

    while (1):
        print("goal_cnt")
        print(goal_cnt)
        # print("gmm_cnt")
        # print(gmm_cnt)
        if (goal_cnt==0 or gmm_cnt==0):
            print("wait for gmm_map and nav_target!")
            rospy.sleep(0.5)
            #geometry_msgs/PoseStamped
        else:
            gmm_nav()
            rospy.sleep(0.5)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    rospy.spin()

if __name__ == "__main__":
    main()