# '''''''''''''
# Author: Dylan
# Date:14/08/2019
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
import time
import datetime

class SoftwareProtecttion(object):
    '''
    About time process class
    '''
    def __init__(self, elapsed_time,elapsed_time_flag=False, mac_protection_flag=False):

        #self.start_time = start_time
        self.elapsed_time = elapsed_time
        self.elapsed_time_flag = elapsed_time_flag
        self.mac_protection_flag = mac_protection_flag
        self.home_path = os.environ['HOME'] + '/' + '.frt'
        self.home_path_2 =os.environ['HOME'] + '/' + '.grt'
        self.cwd = os.getcwd() + '/' + '.crt'

        self.mac_home_path = os.environ['HOME'] + '/' + '.fmac'
        self.mac_home_path_2 = os.environ['HOME'] + '/' + '.gmac'
        self.mac_cwd = os.getcwd() + '/' + '.cmac'
        self.c_time = datetime.datetime.now().replace(microsecond=0)
        self.c_mac = self.get_mac_address()
        if self.elapsed_time_flag == True:
            if not os.path.exists(self.home_path):
                with open(self.home_path, 'a') as f:
                    f.write(str(self.c_time) + '\n') # note use time

            if not os.path.exists(self.home_path_2):
                with open(self.home_path_2, 'a') as f:
                    f.write(str(self.c_time) + '\n') # note use time
            if not os.path.exists(self.cwd):
                with open(self.cwd, 'a') as f:
                    f.write(str(self.c_time) + '\n') # note use time

        if self.mac_protection_flag== True:
            if not os.path.exists(self.mac_home_path):
                with open(self.mac_home_path, 'a') as f:
                    f.write(str(self.c_mac))  # note use time

            if not os.path.exists(self.mac_home_path_2):
                with open(self.mac_home_path_2, 'a') as f:
                    f.write(str(self.c_mac))  # note use time
            if not os.path.exists(self.mac_cwd):
                with open(self.mac_cwd, 'a') as f:
                    f.write(str(self.c_mac))  # note use time

    def is_over_time(self):
        if self.elapsed_time_flag == False and self.mac_protection_flag == False:
            return False
        if self.mac_protection_flag:
            current_txt = self.time_file_handle(False)
            mac_l = ""
            with open(current_txt, 'r') as f:
                mac_l = f.readline()
                #print('mac_l:', mac_l)
            mac_c = self.get_mac_address()
            #print('mac_c:', mac_c)
            if mac_c == mac_l:
                return True

        if self.elapsed_time_flag:
            current_txt = self.time_file_handle(True)
            #start = datetime.datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
            e_time = self.elapsed_time
            use_time = 0
            #print(current_txt)
            with open(current_txt, 'r') as f:
                file = f.readlines()
                first_time = file[0].rstrip('\n')
                #print('first_time:', first_time)
                #print(file)
                latest_time = file[-1].rstrip('\n')
                first_time = datetime.datetime.strptime(first_time, '%Y-%m-%d %H:%M:%S')
                latest_time = datetime.datetime.strptime(latest_time, '%Y-%m-%d %H:%M:%S')
            # 当前时间
            n_time = datetime.datetime.now().replace(microsecond=0)
            #print('n_time:', n_time)
            #print('latest_time:', latest_time)
            # 判断当前时间是否在范围时间内
            if n_time >= latest_time:
                #print('tttt')
                t_time = n_time - first_time
                days = self.calc_days(first_time, n_time)
                #print('days:', days)
                if days <= e_time:
                    self.set_frt(n_time)
                    return False
                return True
            else:
                return True


    def time_file_handle(self, time_or_mac_flag): # time True; mac False
        len1 = 0
        len2 = 0
        len3 = 0
        current_txt = ""
        if time_or_mac_flag:
            txt_list = [self.home_path, self.home_path_2, self.cwd]
            with open(self.home_path, 'r') as f:
                len1 = len(f.readlines())
            with open(self.home_path_2, 'r') as f:
                len2 = len(f.readlines())
            with open(self.cwd, 'r') as f:
                len2 = len(f.readlines())
            if len1 == len2 and len1 == len3:
                current_txt = self.home_path
                return current_txt
        else :
            txt_list = [self.mac_home_path, self.mac_home_path_2, self.mac_cwd]
            with open(self.mac_home_path, 'r') as f:
                len1 = len(f.readlines())
            with open(self.mac_home_path_2, 'r') as f:
                len2 = len(f.readlines())
            with open(self.mac_cwd, 'r') as f:
                len2 = len(f.readlines())
            # if len1 == len2 and len1 == len3:
            #     current_txt = self.home_path
            #     return current_txt
            current_txt = self.mac_cwd
            return current_txt
        len_list = [len1, len2, len3]
        index = len_list.index(max(len_list))
        current_txt = txt_list[index]
        return current_txt

    def set_frt(self, n_time):

        if self.elapsed_time_flag == True:
            with open(self.home_path, 'a') as f:
                f.write(str(n_time) + '\n')  # note use time

            with open(self.home_path_2, 'a') as f:
                f.write(str(n_time) + '\n')  # note use time

            with open(self.cwd, 'a') as f:
                f.write(str(n_time) + '\n')  # note use time

    def get_mac_address(self):
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])

    def calc_days(self, time1, time2):
        year1 = time1.year
        month1 = time1.month
        day1 = time1.day
        year2 = time2.year
        month2 = time2.month
        day2 = time2.day
     #根据year1是否为闰年，选择year1当年每月的天数列表
        if self.isLeapYear(year1):
            daysOfMonths1 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            daysOfMonths1 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    
        year_temp = year2 - year1 
        month_temp = month2 - month1
    
        if year_temp == 0:            #同年
            if month_temp == 0:       #同年且同月
                days = day2 -day1
            else:                     #同年但不同月
                days = daysOfMonths1[month1-1]-day1+day2  #“掐头去尾”
                i = 1 #计算中间月份，若月份相邻则month_temp==1，该循环不会被执行
                while i< month_temp:
                    days += daysOfMonths1[month1+i-1]                
                    i += 1
    
        else:    #不同年
                 #根据是否为闰年，得到year2每月天数列表
            if self.isLeapYear(year2):    
                daysOfMonths2 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            else:
                daysOfMonths2 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            #第一个日期所在年份剩余天数，首先计算第一个月份剩余天数
            days = daysOfMonths1[month1-1]-day1
            i = 1
            while (month1+i<=12):
            #若该月不是12月，即 有剩余月份，则执行该循环，累加本年所剩余天数
                    days += daysOfMonths1[month1+i-1]
                    i += 1
            #计算第二个日期所在年份已经经过的天数
            days += day2#先计算当月已经经过的天数
            if month2 > 1:#若不是1月，则计算已经经过的月份的天数
                j = 1
                while j < month2:
                    days += daysOfMonths2[j-1]
                    j += 1
            #计算中间年份的天数，此时temp_year > 1（即不相邻年份，
            #因为以上的“掐头去尾”已经可以得出年份相邻这种情况的结果了）
            k = 1
            while k < year_temp:
                if self.isLeapYear(year1 + k):
                    days += 366
                else:
                    days += 365
                k += 1
        return(days)   
   
    def isLeapYear(self, year):

        if (year%4 ==0 and year%100 !=0 ) or (year%400 == 0):
            return True
        else:
            return False 

