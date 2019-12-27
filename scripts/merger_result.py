import cv2
import numpy as np
import os
from itertools import combinations,permutations
import  matplotlib.pyplot as plt 

import cv2
import numpy as np
from itertools import combinations,permutations
import  matplotlib.pyplot as plt 
import os


'''
input_image_list = ['./1a_Color.png','./2a_Color.png','./3a_Color.png','./4a_Color.png','./5a_Color.png','./1b_Color.png','./2b_Color.png','./3b_Color.png','./4b_Color.png','./5b_Color.png']
out_final_mask = ['./out_mask1.png','./out_mask2.png']
input_csv_file_list = ['./1a_Color.csv','./2a_Color.csv','./3a_Color.csv','./4a_Color.csv','./5a_Color.csv','./1b_Color.csv','./2b_Color.csv','./3b_Color.csv','./4b_Color.csv','./5b_Color.csv']
'''
def get_final_result_multirows(input_image_list=None,input_csv_file_list=None,out_final_mask=None,merger_file_name=None):

	
	if len(input_image_list)>0:
		index = input_image_list[0].rfind('/')
		result_dir = input_image_list[0][:index+1]
	else:
		log_info = "拼接输入图片个数不正确,输入图片个数应该大于0"
		return log_info 

	if len(input_image_list)%2!=0:
		print("The input image list num is not right")
		log_info = "输入的图片个数不正确,应该输入偶数个图片"
		return log_info
	# ############################
	#算法步骤
	#1.单独处理每一行图片的数据 生成对应的结果数据
	#2.拼接行数据上的两个大图
	#3.处理拼接后的两行的代码
	############################
	pass
	all_img_num = len(input_image_list)
	one_rows_img_num = all_img_num//2

	input_image_list_row1 = input_image_list[::2]
	input_image_list_row2 = input_image_list[1::2]

	input_csv_file_list_row1 = input_csv_file_list[::2]
	input_csv_file_list_row2 = input_csv_file_list[1::2]

	out_final_mask_row1 = out_final_mask[0]
	out_final_mask_row2 = out_final_mask[1]

	print("input_image_list_row1:", input_image_list_row1)
	print("input_image_list_row2:", input_image_list_row2)


	print("input_image_list_row1:", input_image_list_row1)
	print("input_csv_file_list_row1:", input_csv_file_list_row1)
	print("out_final_mask_row1:", out_final_mask_row1)

	#这里分别对每个行的检测结果 
	#行1 融合的csv结果为tmp_merger_final1.csv  融合的掩码名字为tmp_merger_final1.png
	#行2 融合的csv结果为tmp_merger_final2.csv  融合的掩码名字为tmp_merger_final2.png
	log_info = get_final_result(input_image_list=input_image_list_row1,
					 input_csv_file_list=input_csv_file_list_row1,
					 out_final_mask=out_final_mask_row1,
					 merger_file_name='./tmp_merger_final1.csv',
					 save_channel_index=0)
	if(log_info != '正常'):
		return log_info

	log_info = get_final_result(input_image_list=input_image_list_row2,
					input_csv_file_list=input_csv_file_list_row2,
					out_final_mask=out_final_mask_row2,
					merger_file_name='./tmp_merger_final2.csv',
					save_channel_index=1)

	if(log_info != '正常'):
		return log_info

	return '正常'
	############################################################need to do
    #tmp_merger_final1.png tmp_merger_final2.png 这里是生成的掩码图 对应拼接这两个掩码
	###########################这里调用拼接代码
	#
	#生成对应的注释文件
	#将拼接好的行图片移动到对应的文件夹 然后进行拼接
	#
	#
	#
	#
	#
	###################################


	##################################这里得到拼接后的掩码图  名字格式为final_merger_mask.png
	#
	#
	#
	# #输入上面生成的新的掩码图列表
	# input_image_list = ['./tmp_merger_final1.png','./tmp_merger_final1.png']
	# input_csv_file_list = ['./tmp_merger_final1.csv','./tmp_merger_final1.csv']
	# out_final_mask = './final_merger_mask.png'
	# ###################################################上面的参数应该是固定的
	#
	# get_final_result(input_image_list=input_image_list,
	# 				input_csv_file_list=input_csv_file_list,
	# 				out_final_mask=out_final_mask,
	# 				is_row_format=False)


    



#3_Color_mask
#out_mask
#这个文件主要是用来处理拼接后的掩码，使其和原始掩码对应起来,这样就可以找到对应的长轴和短轴i
#这里要做的是将每个图分割开，找到每个如的
#为掩码图中的结果创建一个csv文件方便后面定义
#接口定义
#get_final_result
#input input_image_list:[]凭借输入图片的列表
#	   input_csv_file_list:[] 保存好的csv格式的列表
#	   out_final_mask:[] 拼接好的图片掩码
#return：将合并后结果重新整理到csv文件中，最后根据这个表进行最后处理
def get_final_result(input_image_list=None,input_csv_file_list=None,out_final_mask=None,merger_file_name=None,save_channel_index=0,is_row_format=True):
	
	# creat_match_graph_txt()
	# import pdb
	# pdb.set_trace()
	#这里为了方便测试自己直接付给相应的值
	# test_file_list = ['./demo/1a_Color.png',  './demo/2a_Color.png', './demo/3a_Color.png', './demo/4a_Color.png', './demo/5a_Color.png', 
	# './demo/1b_Color.png','./demo/2b_Color.png',  './demo/3b_Color.png','./demo/4b_Color.png', './demo/5b_Color.png']
	# input_image_list = ['./1a_Color.png',  './2a_Color.png', './3a_Color.png', './4a_Color.png', './5a_Color.png']
	# out_final_mask = './out_mask1.png'
	# # input_csv_file_list = ['./3_Color_5_113.csv','./4_Color_80_198.csv']
	# input_csv_file_list = ['./1a_Color.csv','./2a_Color.csv','./3a_Color.csv','./4a_Color.csv','./5a_Color.csv']

	# input_image_list = ['./1a_Color.png',  './2a_Color.png', './3a_Color.png', './4a_Color.png', './5a_Color.png']
	# out_final_mask = './out_mask1.png'
	# # input_csv_file_list = ['./3_Color_5_113.csv','./4_Color_80_198.csv']
	# input_csv_file_list = ['./1a_Color.csv','./2a_Color.csv','./3a_Color.csv','./4a_Color.csv','./5a_Color.csv']

	# input_image_list = ['./1b_Color.png',  './2b_Color.png', './3b_Color.png', './4b_Color.png', './5b_Color.png']
	# out_final_mask = './out_mask2.png'
	# # input_csv_file_list = ['./3_Color_5_113.csv','./4_Color_80_198.csv']
	# input_csv_file_list = ['./1b_Color.csv','./2b_Color.csv','./3b_Color.csv','./4b_Color.csv','./5b_Color.csv']


	if is_row_format:
		input_image_mask_list = [image_str[:-4]+'_mask.png' for image_str in input_image_list]
		all_final_csv_list = [name[:-4] + '_final.csv' for name in input_csv_file_list]
	else:
		input_image_mask_list = input_image_list
		all_final_csv_list = input_csv_file_list

	print('all_final_csv_list:', all_final_csv_list)
	print('input_image_mask_list:', input_image_mask_list)
	#对于每张图片分开进行处理
	if os.path.exists(out_final_mask):
		img_out_mask =  cv2.imread(out_final_mask)
	else:
		print("img out mask is not exist please check file")
		log_info = "拼接结果掩码图不存在！"
		return log_info
	
	if img_out_mask.shape[0] == 600 and img_out_mask.shape[1] == 400:
		log_info = "拼接结果错误!,拍摄图片重叠面积比较小或者相邻图片之间拍摄角度差别比较大，请按标准重新拍摄"
		return log_info

	if is_row_format:
		[channel_index0,channel_index1,channel_index2] = get_out_mask_split_index(img_out_mask)
	print("check all channel",[channel_index0,channel_index1,channel_index2] )

	if(len(channel_index0)+len(channel_index1)+len(channel_index2)!=len(input_image_list)):
		print("split is wrong!!!! please check")
		log_info = "拼接图片分割结果可能错误！"
		return log_info
	# meger_fin


	# meger_final_result(all_final_csv_list,img_out_mask)

	# import pdb
	# pdb.set_trace()
	
	# ,len(input_image_list)
	for process_index in range(0,len(input_image_list)):
		#假设匹配的点从第0个开始 测试集中就是这样
		#后面根据实际情况 需要添加初始基准的检测算法
		current_strat_index = 0
		current_csv_path = input_csv_file_list[process_index]
		current_mask_path = input_image_mask_list[process_index]

		out_file_csv_path = current_csv_path[:-4]+'_final.csv'

		#读取对应的原始掩码和csv
		if os.path.exists(current_mask_path):
			org_img_mask = cv2.imread(current_mask_path)
		else:
			print("org img mask is not exist")
			log_info = current_mask_path+"，原始检测结果的掩码图片不存在!"
			return log_info
		if os.path.exists(current_csv_path):
			f = open(current_csv_path)
			csv_lines  = f.readlines()
			f.close()
		else:
			print("current csv is not exist")
			log_info = current_csv_path+"，检测结果的csv文件不存在!"
			return log_info

		img_out_mask_cp = img_out_mask.copy()
		#得到两个通道的掩码图
		img_out_m1 = np.array((img_out_mask_cp[:,:,0]>0)*255,dtype=np.uint8)
		img_out_m2 = np.array((img_out_mask_cp[:,:,1]>0)*255,dtype=np.uint8)
		img_out_m3 = np.array((img_out_mask_cp[:,:,2]>0)*255,dtype=np.uint8)
		# cv2.imshow('org2_m1',np.array((img_out_mask_cp[:,:,0]>0)*255,dtype=np.uint8))
		# cv2.imshow('org2_m2',np.array((img_out_mask_cp[:,:,1]>0)*255,dtype=np.uint8))
		# cv2.imshow('org2_m3',np.array((img_out_mask_cp[:,:,2]>0)*255,dtype=np.uint8))
		# cv2.waitKey()
	

		#这里得到最终掩码图上对于当前图片的掩码
		#
		# #######need to continue do some process########
		#
		#
		#
		#现在简单处理
		if is_row_format:
			if(process_index%3==0):
				current_img_index = int(process_index/3)
				start_cols = channel_index0[current_img_index][0]
				end_cols = channel_index0[current_img_index][1]
				img_mask = img_out_m1[:,start_cols:end_cols]
			elif(process_index%3==1):
				current_img_index = int(process_index/3)
				start_cols = channel_index1[current_img_index][0]
				end_cols = channel_index1[current_img_index][1]
				img_mask = img_out_m2[:,start_cols:end_cols]
			else:
				current_img_index = int(process_index/3)
				start_cols = channel_index2[current_img_index][0]
				end_cols = channel_index2[current_img_index][1]
				img_mask = img_out_m3[:,start_cols:end_cols]
			img_out_mask_cp = img_out_mask_cp[:,start_cols:end_cols]
			print(start_cols,end_cols)
		else:
			if(process_index%2==0):
				start_cols = 0
				img_mask = img_out_m1
				img_out_mask_cp = img_out_m1
			else:
				start_cols = 0
				img_mask = img_out_m2
				img_out_mask_cp = img_out_m1

		
		# cv2.imshow('out_mask',img_mask)
		# cv2.waitKey()

		det_point_list = get_sort_box_list_from_mask(img_mask)
		real_point_list = get_sort_box_list_from_csv(csv_lines)

		if(len(det_point_list)!=len(real_point_list)):
			print("box num is not equal")
			continue

		# tmp = img_out_mask_cp.copy()
		# if(process_index%2==0):
			
		# 	tmp[:,:,1] = tmp[:,:,2] = 0
		# else:
		# 	tmp[:,:,0] = tmp[:,:,2] = 0
		# for i in range(len(det_point_list)):
		# 	x,y,x2,y2 = det_point_list[i]
		# 	# current_roi = img_out_mask_cp[y:y2,x:x2,0]
		# 	cv2.rectangle(tmp,(x,y),(x2,y2),(0,0,255),2)
		# 	cv2.imshow('org1',tmp)
		# 	cv2.waitKey()


		# for line in csv_lines:
		# 	# print(line)
		# 	line_s = line.split(',')
		# 	up_x,up_y,down_x,down_y = int(line_s[1]),int(line_s[2]),int(line_s[3]),int(line_s[4])
		# 	cv2.rectangle(org_img_mask,(up_x,up_y),(down_x,down_y),(0,0,255),2)
		# 	cv2.imshow('org1',org_img_mask)
		# 	real_point_list.append([up_x,up_y,down_x,down_y])
		# 	cv2.waitKey()


		point_list = det_point_list

		#匹配的初始化的点
		cur_det_index = 0
		cur_real_index = 0

		adjust_point_list = []
		# adjust_point_list = [point_list[0],point_list[1],point_list[2]]

		init_list  = list([0,1,2,3])
		init_combine = list(combinations(init_list,2))
		print(init_combine)
		print(init_list)

		#初始化相似计算
		#这里找到第一个参考基本点
		real_result = cal_combine_similar(real_point_list,init_combine,init_list)
		print(real_result)	
	
		tmp_org_img_mask_cp = org_img_mask.copy()
		# for i  in range(3):
		# 	# x,y,x2,y2 = init_adjust_list[i]
		# 	# cv2.rectangle(img_out_mask_cp,(x,y),(x2,y2),(0,0,255),2)
		# 	up_x,up_y,down_x,down_y = real_point_list[i]
		# 	cv2.rectangle(tmp_org_img_mask_cp,(up_x,up_y),(down_x,down_y),(0,0,255),2)
		# cv2.imshow('org',tmp_org_img_mask_cp)
		# cv2.waitKey()

		#根据real_list的范围确定初始三个点的范围
		max_x = 0
		max_y = 0
		min_x = 1000000
		min_y = 1000000
		for i  in range(4):
			if(real_point_list[i][0] > max_x):
				max_x = real_point_list[i][0]
			if(real_point_list[i][0] < min_x):
				min_x = real_point_list[i][0]
			if(real_point_list[i][1] > max_y):
				max_y = real_point_list[i][1]
			if(real_point_list[i][1] < min_y):
				min_y = real_point_list[i][1]

		max_x_ratio = float(max_x/org_img_mask.shape[1])
		max_y_ratio = float(max_y/org_img_mask.shape[0])

		min_x_ratio = float(min_x/org_img_mask.shape[1])
		min_y_ratio = float(min_y/org_img_mask.shape[0])

		max_x_range = (max_x_ratio+0.3)*img_out_mask_cp.shape[1]
		max_y_range = (max_y_ratio+0.3)*img_out_mask_cp.shape[0]

		min_x_range = (min_x_ratio-0.3)*img_out_mask_cp.shape[1]
		min_y_range = (min_y_ratio-0.3)*img_out_mask_cp.shape[0]

		print(max_x_ratio,min_x_ratio,max_y_ratio,min_y_ratio)
		print(max_x_range,min_x_range,max_y_range,min_y_range)
		for i in range(10):
			print(point_list[i])
			if(point_list[i][0] <= max_x_range 
				and point_list[i][0] >= min_x_range
				and point_list[i][1] <= max_y_range
				and point_list[i][1] >= min_y_range):
				adjust_point_list.append(point_list[i])
		print(adjust_point_list)

		# det_result = cal_combine_similar(adjust_point_list,init_combine)
		# print(det_result)

		# adjust_point_list = [point_list[2],point_list[1],point_list[0]]
		# det_result = cal_combine_similar(adjust_point_list,init_combine)
		# print(det_result)

		thr = 0.2
		possbile_init_list = []
		init_permutations = list(combinations(range(len(adjust_point_list)),4))
		print(init_permutations)
		for permuta  in init_permutations:
			print(permuta)
			tmp_point_list = list(adjust_point_list)
			tmp_point_list = [tmp_point_list[i] for i in permuta]
			init_permutations_1 = list(permutations(init_list,4))
			for permuta1  in init_permutations_1:
				tmp_point_list1 = list(tmp_point_list)
				tmp_point_list1 = [tmp_point_list1[i] for i in permuta1]
				det_result = cal_combine_similar(tmp_point_list1,init_combine,init_list)
				match=True
				for i in range(len(det_result)):
					if(np.abs(det_result[i]-real_result[i])>thr):
						match=False
				if(match):
					possbile_init_list.append(tmp_point_list1)
					print("possible current_permuta",permuta)
					print(det_result)

					# tmp_out_img_mask_cp = img_out_mask_cp.copy()
					# for i  in range(3):
					# 	up_x,up_y,down_x,down_y = tmp_point_list[i]
					# 	cv2.rectangle(tmp_out_img_mask_cp,(up_x,up_y),(down_x,down_y),(0,0,255),2)
					# cv2.imshow('possible',tmp_out_img_mask_cp)
					# cv2.waitKey()

		# 在我们测试用例前三个点是匹配的 debug代码后面可以注释
		
		#尝试进行匹配
		print('len:',len(possbile_init_list))
		for init_adjust_list  in possbile_init_list:

			org_img_mask_cp = org_img_mask.copy()
			img_out_mask_cp1 = img_out_mask_cp.copy()
			for i  in range(4):
				x,y,x2,y2 = init_adjust_list[i]
				cv2.rectangle(img_out_mask_cp1,(x,y),(x2,y2),(0,0,255),2)
				up_x,up_y,down_x,down_y = real_point_list[i]

				print('init real:',up_x,up_y,down_x,down_y ,'init det:',x,y,x2,y2)
				cv2.rectangle(org_img_mask_cp,(up_x,up_y),(down_x,down_y),(0,0,255),2)

			# cv2.imshow('org',org_img_mask_cp)
			# cv2.imshow('out',img_out_mask_cp1)
			# cv2.waitKey(2)
			print("try match",'='*100)
			copy_point_list = list(point_list)
			for point in init_adjust_list:
				copy_point_list.remove(point)
			re_adjust_points_list = get_adjust_point_list(init_adjust_list,list(real_point_list),list(copy_point_list),
													cur_real_index=4,cur_det_index=0,
													org_img = org_img_mask_cp.copy(),img = img_out_mask_cp1.copy())
			print(len(re_adjust_points_list),len(real_point_list[current_strat_index:]))
			if len(re_adjust_points_list)==len(real_point_list[current_strat_index:]):
				print("matched box num is:",len(re_adjust_points_list),'current_strat_index:',current_strat_index)
				#保存最终结果
				save_out_mask_csv_file(re_adjust_points_list,csv_lines,current_strat_index,out_file_csv_path,start_cols)
				#
				break
		# cv2.waitKey()

	print(all_final_csv_list)	
	
	meger_final_result(all_final_csv_list,img_out_mask,merger_file_name,save_channel_index)
	return '正常'

def cal_iou(box1,box2):
	
	w = min(box1[2],box2[2]) - max(box1[0],box2[0])
	h = min(box1[3],box2[3]) - max(box1[1],box2[1])

	if(w<=0 or h<= 0):
		return 0
	current_area = (box1[2]-box1[0])*(box1[3]-box1[1])
	next_area = (box2[2]-box2[0])*(box2[3]-box2[1])
	cross = w*h
	return cross/(current_area+next_area-cross)


'''

'''
def meger_final_result(all_final_csv_list,out_image_mask,meger_final_name=None,channel_index=0):
	
	if  meger_final_name:
		meger_final_name = meger_final_name
	else:
		meger_final_name = './meger_final.csv'

	merger_final_mask_name = meger_final_name[:-4]+'.png'
	
	meger_final_result_list = []

	img_out_mask_cp = out_image_mask.copy()
	merger_final_mask = np.zeros(shape=img_out_mask_cp.shape)
	all_lines_list = []
	for i in range(len(all_final_csv_list)):
		if(not os.path.exists(all_final_csv_list[i])):
			continue
		f =open(all_final_csv_list[i],'r')
		all_lines = f.readlines()
		f.close()
		all_lines_list.append(all_lines)

	for i in range(len(all_lines_list)):
		current_lines = all_lines_list[i]
		for line in current_lines:
			line_s = line.split(',')
			x,y,x2,y2 = int(line_s[-4]),int(line_s[-3]),int(line_s[-2]),int(line_s[-1])
			print('init det:',x,y,x2,y2)
			cv2.rectangle(img_out_mask_cp,(x,y),(x2,y2),(0,0,255),2)
			#cv2.imshow('final mask',img_out_mask_cp)
			#cv2.waitKey()
			# 有重叠区域
			current_channel = -1
			exist_channel = -1
			if((np.sum(out_image_mask[y:y2,x:x2,0])>25 
				and np.sum(out_image_mask[y:y2,x:x2,1])>25)):
				current_channel=0
				exist_channel = 1
			elif(np.sum(out_image_mask[y:y2,x:x2,1])>25 
				and np.sum(out_image_mask[y:y2,x:x2,2])>25):
				current_channel=1
				exist_channel = 2
			elif(np.sum(out_image_mask[y:y2,x:x2,2])>25 
				and np.sum(out_image_mask[y:y2,x:x2,0])>25):
				current_channel=2
				exist_channel = 0
			if exist_channel != -1:
				print("this box is exits in next img")
				if(i+1<len(all_lines_list)):
					next_lines = all_lines_list[i+1]
				else:
					next_lines = []
				for next_line in next_lines:
					next_line_s = next_line.split(',')
					n_x,n_y,n_x2,n_y2 = int(next_line_s[-4]),int(next_line_s[-3]),int(next_line_s[-2]),int(next_line_s[-1])

					if(cal_iou([x,y,x2,y2],[n_x,n_y,n_x2,n_y2])>0.4):
						print("ok,find corr box in next img")
						current_area = (x2-x)*(y2-y)
						next_area = (n_x2-n_x)*(n_y2-n_y)
						if(next_area>current_area):
							meger_final_result_list.append(next_line)	
							merger_final_mask[n_y:n_y2,n_x:n_x2,channel_index] = out_image_mask[n_y:n_y2,n_x:n_x2,exist_channel]
						else:				
							meger_final_result_list.append(line)
							merger_final_mask[y:y2,x:x2,channel_index] = out_image_mask[y:y2,x:x2,current_channel]
						next_lines.remove(next_line)
						break

			else:
				meger_final_result_list.append(line)
				merger_final_mask[y:y2,x:x2,channel_index] = out_image_mask[y:y2,x:x2,0]+out_image_mask[y:y2,x:x2,1]+out_image_mask[y:y2,x:x2,2]
	f = open(meger_final_name,'w')
	for line in meger_final_result_list:
		f.write(line)
	f.close()

	merger_final_mask = np.array(merger_final_mask,dtype=np.uint8)
	cv2.imwrite(merger_final_mask_name,merger_final_mask)





#这个函数用来对应的匹配文件
txt_info='''{{center_image_index | {0} | center image index}}
{{center_image_rotation_angle | {1} | center image rotation angle}}
{{images_count | {2} | images count}}
{3}'''
def creat_match_graph_txt(img_num=5,file_name=None):
# {matching_graph_image_edges-0 | 1 | matching graph image edge 0}
# {matching_graph_image_edges-1 | 0,2 | matching graph image edge 1}
# {matching_graph_image_edges-2 | 1,3 | matching graph image edge 2}
# {matching_graph_image_edges-3 | 2,4 | matching graph image edge 3}
# {matching_graph_image_edges-4 | 3 | matching graph image edge 4}
	one_item_info_tm = '{{matching_graph_image_edges-{} | {} | matching graph image edge {}}}\n'
	all_item_info =''
	for i in range(img_num):
		if(i==0):
			one_item_info = one_item_info_tm.format(0,1,0)
		elif(i==img_num-1):
			one_item_info = one_item_info_tm.format(i,i-1,i)
		else:
			one_item_info = one_item_info_tm.format(i,str(i-1),i)
		all_item_info = all_item_info + one_item_info

	final_txt = txt_info.format(img_num//2,0,img_num,all_item_info)
	print(final_txt)

# 从掩码图上得到每个图的分割点
# 现在使用三个通道的情况
def get_out_mask_split_index(img_out_mask):
	#得到每张图上对应的掩码
	img_out_mask_cp = img_out_mask.copy()
	#得到两个通道的掩码图
	img_out_m1 = np.array((img_out_mask_cp[:,:,0]>0)*255,dtype=np.uint8)
	img_out_m2 = np.array((img_out_mask_cp[:,:,1]>0)*255,dtype=np.uint8)
	img_out_m3 = np.array((img_out_mask_cp[:,:,2]>0)*255,dtype=np.uint8)
	img_h = img_out_mask_cp.shape[0]
	img_w = img_out_mask_cp.shape[1]
	img_out_line1 = np.sum(img_out_m1,axis=0)
	img_out_line2 = np.sum(img_out_m2,axis=0)
	img_out_line3 = np.sum(img_out_m3,axis=0)
	# plt.figure()
	# plt.plot(img_out_line1)
	# plt.plot(img_out_line2)
	# plt.plot(img_out_line3)
	# plt.show()
	img1_zero_index = np.where(img_out_line1<=10)[0]
	img2_zero_index = np.where(img_out_line2<=10)[0]
	img3_zero_index = np.where(img_out_line3<=10)[0]
	def get_split_index(img1_zero_index,img_out_line1):
		# print(img1_zero_index)
		split_index1 = []
		
		for i in range(len(img1_zero_index)):
			current_zeros_index = img1_zero_index[i]
			start_index = current_zeros_index-5 if current_zeros_index-5>=0 else 0
			end_index = current_zeros_index+5 if current_zeros_index+5<len(img_out_line1) else len(img_out_line1)-1
			if(sum(img_out_line1[start_index:end_index]) < 500):
				# print(current_zeros_index)
				split_index1.append(current_zeros_index)
			if(len(split_index1)>=2 and (current_zeros_index-split_index1[-2]<10)):	
				# print(current_zeros_index)
				split_index1.remove(split_index1[-2])
		# 	print(split_index1)
		print(split_index1)
		new_index_pair = []
		for index in range(len(split_index1)-1):
			if index==0:
				current_start_index =0
				current_end_index = split_index1[index]
				if(sum(img_out_line1[current_start_index:current_end_index])>2000):
					new_index_pair.append([current_start_index,current_end_index])
			current_start_index = split_index1[index]
			current_end_index = split_index1[index+1]
			if(sum(img_out_line1[current_start_index:current_end_index])>2000):
				new_index_pair.append([current_start_index,current_end_index])
		return new_index_pair
	# print(img1_zero_index)
	# print(img2_zero_index)

	split_index1 = get_split_index(img1_zero_index,img_out_line1)
	split_index2 = get_split_index(img2_zero_index,img_out_line2)
	split_index3 = get_split_index(img3_zero_index,img_out_line3)
	
	print(split_index1)
	print(split_index2)
	print(split_index3)

	return [split_index1,split_index2,split_index3]


#将已经计算好的数据从新写到对于的out_mask文件夹中
def save_out_mask_csv_file(out_mask_bbox_list,real_csv_lines,current_strat_index,out_file_path,start_cols):
	f=open(out_file_path,'w')
	for i in range(len(out_mask_bbox_list)):
		# f.write()
		if(len(out_mask_bbox_list[i])>0):
			x,y,x2,y2 = out_mask_bbox_list[i]
		else:
			x,y,x2,y2 = 0,0,0,0

		current_line = real_csv_lines[i+current_strat_index]
		log_info = "{},{},{},{},{}\n".format(current_line[:-1],x+start_cols,y,x2+start_cols,y2)
		print(log_info)
		f.write(log_info)
	f.close()


#从当前的掩码图上检测矩形轮廓
#返回检测的box列表
def get_sort_box_list_from_mask(img_mask):

	org_m = img_mask.copy()
	cnts,_ = cv2.findContours(org_m.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	up_left_point_list = []
	point_list = []
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)	
		up_left_point_list.append([y,x])
		point_list.append([x,y,x+w,y+h])
	sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
	print(sort_index)
	# print(type(sort_index))
	#sort_index = np.array(sort_index,dtype=np.int)
	# sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
	point_list = [point_list[i] for i in sort_index]

	print("all detect box num is",len(point_list))
	return point_list

#从原始的csv文件中读取对应的box列表
def get_sort_box_list_from_csv(csv_lines):
	real_point_list = []
	
	for line in csv_lines:
		# print(line)
		line_s = line.split(',')
		up_x,up_y,down_x,down_y = int(line_s[1]),int(line_s[2]),int(line_s[3]),int(line_s[4])
		# cv2.rectangle(org_img,(up_x,up_y),(down_x,down_y),(0,0,255),2)
		# cv2.imshow('org1',org_img)
		real_point_list.append([up_x,up_y,down_x,down_y])
		cv2.waitKey(2)
	print("all real detect box num is",len(csv_lines))
	return real_point_list


# org_out_img = cv2.imread('./out.png')
# org_img = cv2.imread('./3_Color_mask.png')
# img = cv2.imread('./out_mask.png')

# # 3_Color_5_113
# # 4_Color_80_198
# detect_path = './3_Color_5_113.csv'

# f = open(detect_path)
# csv_lines  = f.readlines()
# f.close()
# # cv2.imshow('org_o_img',org_out_img[20:100,260:320])
# # cv2.imshow('org1',org_img)

# # for i in range(0,360):
# # 	print(i,'==',np.cos(i/180*3.14))


# org2_m1 = np.array((img[:,:,0]>0)*255,dtype=np.uint8)
# org2_m2 = np.array((img[:,:,1]>0)*255,dtype=np.uint8)
# cv2.imshow('org2_m1',np.array((img[:,:,0]>0)*255,dtype=np.uint8))
# cv2.imshow('org2_m2',np.array((img[:,:,1]>0)*255,dtype=np.uint8))

# cnts,_ = cv2.findContours(org2_m1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# up_left_point_list = []
# point_list = []
# for c in cnts:
# 	x,y,w,h = cv2.boundingRect(c)
	
# 	up_left_point_list.append([y,x])
# 	point_list.append([x,y,x+w,y+h])
# sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
# print(sort_index)
# # print(type(sort_index))
# #sort_index = np.array(sort_index,dtype=np.int)
# # sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
# point_list = [point_list[i] for i in sort_index]


# print("all detect box num is",len(point_list))
# for i in range(len(point_list)):
# 	x,y,x2,y2 = point_list[i]
# 	current_roi = img[y:y2,x:x2,0]
# 	# cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)
# 	# index = (current_roi != 0)
# 	# mean_val= np.mean(current_roi[index])
# 	# print(mean_val)
# 	# cv2.imshow('org1',img)
# 	# cv2.waitKey()


# real_point_list = []
# print("all real detect box num is",len(csv_lines))
# for line in csv_lines:
# 	# print(line)
# 	line_s = line.split(',')
# 	up_x,up_y,down_x,down_y = int(line_s[1]),int(line_s[2]),int(line_s[3]),int(line_s[4])
# 	# cv2.rectangle(org_img,(up_x,up_y),(down_x,down_y),(0,0,255),2)
# 	# cv2.imshow('org1',org_img)
# 	real_point_list.append([up_x,up_y,down_x,down_y])
# 	# cv2.waitKey()


# cur_det_index = 0
# cur_real_index = 0

# adjust_point_list = []


# adjust_point_list = [point_list[0],point_list[1],point_list[2]]



# init_list  = list([0,1,2])
# init_combine = list(combinations(init_list,2))

# print(init_combine)


def cal_vector_dis(point):
   	p1_x,p1_y = point
   	dis = np.sqrt(p1_x*p1_x+p1_y*p1_y)
   	return dis

# 0.01 的差距9度左右
def cal_cos_similar(point1,point2):
	p1_x,p1_y = point1
	p2_x,p2_y = point2
	# print(point1,point2)
	cos_distance = (p1_x*p2_x+p1_y*p2_y)/(np.sqrt(p1_x*p1_x+p1_y*p1_y)*np.sqrt(p2_x*p2_x+p2_y*p2_y))
	# print(cos_distance)
	return cos_distance


# 计算一个点的集合的组合的相似度，这里点的集合为3，用于找到初始的基准匹配点
def cal_combine_similar(point_list,combines,init_list):
	# print(init_list)
	cal_cos_result = []
	for combine in combines:
		index1 = combine[0]
		index2 = combine[1]

		init_point_index = -1
		for tmp_index in init_list:
			if tmp_index not in combine:
				init_point_index = tmp_index

				# print("init_point index is:",init_point_index,",combinations is:",index1,index2)

				init_box = point_list[init_point_index]
				init_point = [init_box[0],init_box[1]]			

				current_box1 = point_list[index1]
				current_box2 = point_list[index2]
				current_point1 = [current_box1[0]-init_box[0],current_box1[1]-init_box[1]]
				current_point2 = [current_box2[0]-init_box[0],current_box2[1]-init_box[1]]
				cos_sim = cal_cos_similar(current_point1,current_point2)
				cal_cos_result.append(cos_sim)
	return cal_cos_result

# #这里找到第一个参考基本点
# real_result = cal_combine_similar(real_point_list,init_combine)
# print(real_result)
# det_result = cal_combine_similar(adjust_point_list,init_combine)
# print(det_result)

# # adjust_point_list = [point_list[2],point_list[1],point_list[0]]

# # det_result = cal_combine_similar(adjust_point_list,init_combine)
# # print(det_result)


# thr = 0.02
# possbile_init_list = []
# init_permutations = list(permutations(init_list,3))
# for permuta  in init_permutations:
# 	print("current_permuta",permuta)
# 	tmp_point_list = list(adjust_point_list)
# 	tmp_point_list = [tmp_point_list[i] for i in permuta]
# 	det_result = cal_combine_similar(tmp_point_list,init_combine)
# 	if((np.abs(det_result[0]-real_result[0])<thr) 
# 		and (np.abs(det_result[1]-real_result[1])<thr) 
# 		and (np.abs(det_result[2]-real_result[2])<thr)):
# 		possbile_init_list.append(tmp_point_list)
# 	print(det_result)
# #在我们测试用例前三个点是匹配的
# for i  in range(3):
# 	x,y,x2,y2 = point_list[i]
# 	cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)
# 	up_x,up_y,down_x,down_y = real_point_list[i]
# 	cv2.rectangle(org_img,(up_x,up_y),(down_x,down_y),(0,0,255),2)

# cv2.imshow('org1',org_img)
# cv2.imshow('out',img)
# cv2.waitKey()


#从box的列表中得到对应index的左上角的坐标
#input:
#return:如果当前列表元素为空则返回空列表
def get_corner_point(box_list,index):
	if(index>=len(box_list)):
		return []
	current_box = box_list[index]
	
	if(len(current_box)>0):
		return [current_box[0],current_box[1]]
	else:
		return []

#从最近的三个点中生成三个方向向量，用于后边的相似性判断
#input:
#return:
def create_dir_vector(current_real_point,l_real_point,ll_real_point):
	
	#得到历史列表中最近的一个轨迹向量以及当前点到该向量两个端点的向量
	last_line = [l_real_point[0] -ll_real_point[0],l_real_point[1]-ll_real_point[1]]
	c_l_line = [current_real_point[0]-l_real_point[0],current_real_point[1]-l_real_point[1]]
	c_ll_line = [current_real_point[0] - ll_real_point[0],current_real_point[1]-ll_real_point[1]]

	return c_l_line,c_ll_line,last_line

#从循环检测的列表中得到最有可能的下个点的索引值
def get_most_possible_index(find_det_index_list):
	count = np.zeros(10)
	for i in find_det_index_list:
		count[i] = count[i]+1
	return np.argmax(count)


#从一个初始的状态中找到最终的点的匹配过程
def get_adjust_point_list(init_adjust_list,real_point_list,point_list,cur_real_index=3,cur_det_index=3,org_img=None,img=None):

	adjust_point_list = list(init_adjust_list)
	#这里可能存在的情况是在于一开始对应的存在于原始的中间
	#原始图上的index可能要重新找一些基准匹配点 初始化一定三个点
	cur_real_index = cur_real_index
	cur_det_index = cur_det_index

	not_find_num = 0
	while(cur_real_index<len(real_point_list)):

		current_real_point = get_corner_point(real_point_list,cur_real_index)
		#找到最近的三个有效的点防止形变带来的误差
		
		init_combine = list(combinations(range(len(adjust_point_list)),3))
		random_index = np.random.choice(len(init_combine),len(init_combine),replace=False)
		init_combine = [init_combine[i] for i in random_index]
		count = 0
		find_det_index_list = []
		for combine in init_combine:
			# print(combine)
			hist_real_point = []
			hist_det_point = []
			det_point1 = get_corner_point(adjust_point_list,combine[0])
			det_point2 = get_corner_point(adjust_point_list,combine[1])
			det_point3 = get_corner_point(adjust_point_list,combine[2])
			if(len(det_point1)>0 and len(det_point2)>0 and len(det_point3)>0):

				if(count>30):
				 	break
				count = count+1
				hist_det_point.append(det_point1)
				hist_det_point.append(det_point2)
				hist_det_point.append(det_point3)
				
				real_point1 = get_corner_point(real_point_list,cur_real_index-len(adjust_point_list)+combine[0] )
				real_point2 = get_corner_point(real_point_list,cur_real_index-len(adjust_point_list)+combine[1] )
				real_point3 = get_corner_point(real_point_list,cur_real_index-len(adjust_point_list)+combine[2] )
				hist_real_point.append(real_point1)
				hist_real_point.append(real_point2)
				hist_real_point.append(real_point3)
		
				#
				c_l_real_line,c_ll_real_line,last_real_line=create_dir_vector(current_real_point,hist_real_point[0],hist_real_point[1])
				c_l_real_line1,c_ll_real_line1,last_real_line1=create_dir_vector(current_real_point,hist_real_point[1],hist_real_point[2])
				#计算四个相似度
				cos_sim1 = cal_cos_similar(c_l_real_line,last_real_line)
				cos_sim2 = cal_cos_similar(c_ll_real_line,last_real_line)
				cos_sim3 = cal_cos_similar(c_l_real_line1,last_real_line1)
				cos_sim4 = cal_cos_similar(c_ll_real_line1,last_real_line1)

				last_real_line_lenght = cal_vector_dis(last_real_line)
				last_real_line1_lenght = cal_vector_dis(last_real_line1)
				c_l_real_line_lenght = cal_vector_dis(c_l_real_line)
				c_l_real_line1_lenght = cal_vector_dis(c_l_real_line1)

				real_ratio1 = c_l_real_line_lenght/last_real_line_lenght
				real_ratio2 = c_l_real_line1_lenght/last_real_line1_lenght

				print("DEBUG:real_cos_sim:",cos_sim1,cos_sim2,cos_sim3,cos_sim4,real_ratio1,real_ratio2)
				up_x,up_y,down_x,down_y = real_point_list[cur_real_index]
				cv2.rectangle(org_img,(up_x,up_y),(down_x,down_y),(0,0,255),2)
				# cv2.imshow('org1',org_img)
				# cv2.waitKey()

				#在最近的点中搜索与目标相似度最接近的点
				find = False
				best_cos_sim1 = 100000000
				best_cos_sim2 = 100000000
				best_cos_sim3 = 100000000
				best_cos_sim4 = 100000000
				best_real_ratio1 = 100000000
				best_real_ratio2 = 100000000
				final_index = -1
				for i in range(10):
					tmp_det_index = int(cur_det_index)
					if((tmp_det_index+i)>=len(point_list)):
						break
					current_det_point = get_corner_point(point_list,tmp_det_index+i)
					
					# print(hist_det_point)
					c_l_det_line,c_ll_det_line,last_det_line=create_dir_vector(current_det_point,hist_det_point[0],hist_det_point[1])
					c_l_det_line1,c_ll_det_line1,last_det_line1=create_dir_vector(current_det_point,hist_det_point[1],hist_det_point[2])

					cos_det_sim1 = cal_cos_similar(c_l_det_line,last_det_line)
					cos_det_sim2 = cal_cos_similar(c_ll_det_line,last_det_line)

					cos_det_sim3 = cal_cos_similar(c_l_det_line1,last_det_line1)
					cos_det_sim4 = cal_cos_similar(c_ll_det_line1,last_det_line1)



					last_det_line_lenght = cal_vector_dis(last_det_line)
					last_det_line1_lenght = cal_vector_dis(last_det_line1)
					c_l_det_line_lenght = cal_vector_dis(c_l_det_line)
					c_l_det_line1_lenght = cal_vector_dis(c_l_det_line1)

					det_ratio1 = c_l_det_line_lenght/last_det_line_lenght
					det_ratio2 = c_l_det_line1_lenght/last_det_line1_lenght


					print("DEBUG:det_cos_sim,test:",i,cos_det_sim1,cos_det_sim2,cos_det_sim3,cos_det_sim4,det_ratio1,det_ratio2)

					tmp_img = img.copy()
					x,y,x2,y2 = point_list[i+tmp_det_index]
					# cv2.rectangle(tmp_img,(x,y),(x2,y2),(0,0,255),2)
					# cv2.imshow('tmp_out',tmp_img)
					# cv2.waitKey()

					thr = 0.6

					s1 = np.abs(cos_det_sim1-cos_sim1)
					s2 = np.abs(cos_det_sim2-cos_sim2)
					s3 = np.abs(cos_det_sim3-cos_sim3)
					s4 = np.abs(cos_det_sim4-cos_sim4)
					r1 = np.abs(real_ratio1-det_ratio1)
					r2 = np.abs(real_ratio2-det_ratio2)

					match_num = (int(np.abs(cos_det_sim1-cos_sim1)<thr) 
						+ int(np.abs(cos_det_sim2-cos_sim2)<thr) 
						+ int(np.abs(cos_det_sim3-cos_sim3)<thr) 
						+ int(np.abs(cos_det_sim4-cos_sim4)<thr))
					dir_match = ((cos_det_sim1*cos_sim1)>=0
								and(cos_det_sim2*cos_sim2)>=0
								and(cos_det_sim3*cos_sim3)>=0
								and(cos_det_sim4*cos_sim4)>=0
								)
					dis_match = ((np.abs(real_ratio1-det_ratio1)<0.7) 
								and  (np.abs(real_ratio2-det_ratio2)<0.7) 
								)
					
					if(match_num>=4 and dir_match and dis_match):
						find = True
					else:
						find = False
						# print('match num is',match_num)

					if(find and (r1<best_real_ratio1 and  r2<best_real_ratio2)
						and (s1-best_cos_sim1)<0.03
						and (s2-best_cos_sim2)<0.03
						and (s3-best_cos_sim3)<0.03
						and (s4-best_cos_sim4)<0.03):


						best_real_ratio2 = r2
						best_real_ratio1 = r1
						best_cos_sim1 = s1
						best_cos_sim2 = s2
						best_cos_sim3 = s3
						best_cos_sim4 = s4
						final_index = i
						
						#cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)
						#添加该点调整后的点的列表
						# adjust_point_list.append(point_list[tmp_det_index+i])
						# if(i==0):
						# 	tmp_det_index = tmp_det_index +1
						# else:
						# 	point_list.remove(point_list[tmp_det_index+i])
						
				if(not find):	
					pass
					#加入一个空的展位点
					#adjust_point_list.append([])
				if(final_index != -1):
					find_det_index_list.append(final_index)
		if(len(init_combine)>=10): 
			thr_num = 3
		else:
			thr_num = 0
		if(len(find_det_index_list)>thr_num):
			
			print("great find match box!!!!!!!!!!!!!!!!!!!!!!!")
			next_index = get_most_possible_index(find_det_index_list)
			print(find_det_index_list,next_index)	
			x,y,x2,y2 = point_list[next_index+cur_det_index]
			cv2.rectangle(img,(x,y),(x2,y2),(0,0,255),2)
			adjust_point_list.append(point_list[cur_det_index+next_index])
			if(next_index==0):		
				#添加该点调整后的点的列表		
				cur_det_index = cur_det_index +1 
			else:
				point_list.remove(point_list[cur_det_index+next_index])

		else:
			adjust_point_list.append([])
			not_find_num = not_find_num + 1
		
		#有5个点以上的为匹配说明未找到
		if(not_find_num >=2):
			print("ERRORRRRRRRRRR!!!!!!!!!!!!!!!!!!,can not match,THE init status is WRONG!!!!!!!!!!!")
			return []

		# cv2.imshow('org1',org_img)
		# cv2.imshow('img',img)
		# cv2.waitKey(1)
			
		cur_real_index = cur_real_index + 1
		print(cur_real_index,cur_det_index)
	#遍历结束找到所有点
	return adjust_point_list


# print('len:',len(possbile_init_list))
# for init_adjust_list  in possbile_init_list:
# 	print('='*100)
# 	get_adjust_point_list(init_adjust_list,list(real_point_list),list(point_list),cur_real_index=3,cur_det_index=3,org_img = org_img.copy(),img = img.copy())

# cv2.waitKey()




if __name__ == "__main__":
	# get_final_result()
    get_final_result_multirows(input_image_list=None,input_csv_file_list=None,out_final_mask=None,merger_file_name=None)




###################################################################################################old way using pixel val:result is not good
# kernel = np.eye(15,dtype=np.uint8)
# kernel = np.array((kernel + np.rot90(kernel))>0,dtype=np.uint8)
# kernel[:,7]=1
# print(kernel)
# org2_m1_er = cv2.erode(org2_m1,kernel,iterations=1)
# org2_m2_er = cv2.erode(org2_m2,kernel,iterations=1)

# cv2.imshow('org2_m1_er',org2_m1_er)
# cv2.imshow('org2_m2_er',org2_m2_er)

# for i in range(20,100):
# 	for j in range(260,320):
# 		print(img[i,j,0],end =' ')
# 	print('\n')
# print("=============================================================================")
# for i in range(20,100):
# 	for j in range(260,320):
# 		print(img[i,j,1],end = ' ')
# 	print('\n')

# print(img.shape)


# cv2.waitKey()

'''
min_val = np.min(img)
max_val = np.max(img)
print(np.max(img))
print(np.min(img))

def get_mask_code(mask_index_val):
	tmp = int(mask_index_val)
	result_n = []
	while(tmp!=0):
	    n = tmp % 25
	    result_n.append(n*10)
	    tmp = int(tmp/25)


	for i in range(len(result_n),3):
	    result_n.append(0)
	return result_n


for i in range(5,255,2):
	# print(i)
	# result_n = get_mask_code(i)
	# print(result_n)
	result_n = [i,i,0]
	result = ((org_img[:,:,0]-result_n[0]<2))#(org_img[:,:,1]-result_n[1]<2)*(org_img[:,:,2]-result_n[2]<2))
	if(np.sum(result)>0):
		print(org_img[result])
		print(np.max(org_img[result,0]-result_n[0]),np.max(org_img[result,1]-result_n[1]),np.max(org_img[result,2]-result_n[2]))
	result = np.array(result*255,dtype=np.uint8)
	cv2.imshow('org',result)

	result = ((img[:,:,0]-result_n[0]<2))#*(img[:,:,1]-result_n[1]<2)*(img[:,:,2]-result_n[2]<2))
	if(np.sum(result)>0):
		print(img[result])
		print(np.max(img[result,0]-result_n[0]),np.max(img[result,1]-result_n[1]),np.max(img[result,2]-result_n[2]))
	result = np.array(result*255,dtype=np.uint8)
	cv2.imshow('mask',result)
	cv2.waitKey()

	



cv2.imshow('mask',img)

cv2.waitKey()

'''
