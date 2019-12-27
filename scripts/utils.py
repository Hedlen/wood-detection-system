import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import config
if not config.DEBUG:
    # from woodev.apis import init_detector, inference_detector
    import pycocotools.mask as maskUtils
    import mmcv

from PIL import Image
import cv2
import numpy as np

# 这个函数是从 mmdet.apis 的interface 代码中复制过来的,这里修改使其返回图片，而不是保存

# TODO: merge this method with the one in BaseDetector

def new_show_result(img, result, class_names, score_thr=0.3, out_file=None):

    """Visualize the detection results on the image.



    Args:

        img (str or np.ndarray): Image filename or loaded image.

        result (tuple[list] or list): The detection result, can be either

            (bbox, segm) or just bbox.

        class_names (list[str] or tuple[str]): A list of class names.

        score_thr (float): The threshold to visualize the bboxes and masks.

        out_file (str, optional): If specified, the visualization result will

            be written to the out file instead of shown in a window.

    """

    assert isinstance(class_names, (tuple, list))

    img = mmcv.imread(img)

    if isinstance(result, tuple):

        bbox_result, segm_result = result

    else:

        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)



    labels = [

        np.full(bbox.shape[0], i, dtype=np.int32)

        for i, bbox in enumerate(bbox_result)

    ]

    labels = np.concatenate(labels)

    # print(labels)

    # draw segmentation masks

    bbox_color = (0, 255, 0)

    pixels_output = []

    if segm_result is not None:

        segms = mmcv.concat_list(segm_result)

        inds = np.where(bboxes[:, -1] > score_thr)[0]

        for i in inds:
            
            label = labels[i]
            if label > 1:
                continue

            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

            mask = maskUtils.decode(segms[i]).astype(np.bool)

            img[mask] = img[mask] * 0.5 + color_mask * 0.5

            bbox_int = bboxes[i].astype(np.int32)

            left_top = (bbox_int[0], bbox_int[1])

            right_bottom = (bbox_int[2], bbox_int[3])

            cv2.rectangle(

                img, left_top, right_bottom, bbox_color, thickness=1)

            #print('label_names:', class_names)

            label_loc = (bbox_int[0]//2 + bbox_int[2]//2 - 30, bbox_int[1]//2 + bbox_int[3]//2 - 8)

            label_pixels = (bbox_int[0]//2 + bbox_int[2]//2 - 20, bbox_int[1]//2 + bbox_int[3]//2 + 8)


            label_text = class_names[

                label] if class_names is not None else 'cls {} {}'.format(label, i)

            label_text = label_text + ' ' + str(i + 1)

            cv2.putText(img, label_text, label_loc,

                        cv2.FONT_HERSHEY_COMPLEX,  0.5, (255, 0, 0), 1)

            pixels = len(mask[mask == True])

            pixels_text = label_text + ':{}'.format(pixels)

            #cv2.putText(img, pixels_txt, label_pixels,

                        #cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0))

        #label_text = 'num|{:d}'.format(len(inds))

        #cv2.putText(img, label_text, (20, 20),

        #        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

            pixels_output.append(pixels_text)

    img = Image.fromarray(np.uint8(img))

    if out_file==None:

        #img.show()

        return img, inds, pixels_output

    else:

        img.save(out_file)

        cv2.waitKey(0)

        dir(img)

#通过文件路径得到原始深度图的数据
#这里文件的深度名字格式固定可以从原始图的名字中解析出来
def get_depth_data(file_path):
    f = open(file_path,'rb')
    raw_data = f.read()
    f.close()
    depth_data = []
    for i in range(int(len(raw_data)/2)):
        depth_data.append(int.from_bytes(raw_data[i*2:i*2+2],byteorder='little'))
    depth_data = np.array(depth_data)
    depth_data =  np.resize(depth_data, (360,640))
    depth_data = depth_data.transpose([1,0])
    return depth_data

# 得到每个对应的椭圆参数
# input:一个分割掩码的二值图
# return:list [椭圆对象,短轴线段，长轴线段]
def get_ellipse(mask):
    pass
    mask = np.array(mask,dtype=np.uint8)
    #找到边界点来拟合
    contous,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts = contous[0]
    #根据边界点来拟合椭圆
    ellipse_res = cv2.fitEllipse(cnts)
    #得到椭圆的长轴和短轴
    center_p = ellipse_res[0]
    center_x =  center_p[0]
    center_y =  center_p[1]
    axisa = ellipse_res[1][0]
    axisb = ellipse_res[1][1]
    r_angel = ellipse_res[2]
    theta = r_angel * 3.14/180

    a_p1_x = int(center_x + axisa*np.cos(theta)/2)
    a_p1_y = int(center_y + axisa*np.sin(theta)/2)
    a_p2_x = int(center_x - axisa*np.cos(theta)/2)
    a_p2_y = int(center_y - axisa*np.sin(theta)/2)

    b_p1_x = int(center_x + axisb*np.cos(theta+3.14/2)/2)
    b_p1_y = int(center_y + axisb*np.sin(theta+3.14/2)/2)
    b_p2_x = int(center_x - axisb*np.cos(theta+3.14/2)/2)
    b_p2_y = int(center_y - axisb*np.sin(theta+3.14/2)/2)

    line_a = [a_p1_x,a_p1_y,a_p2_x,a_p2_y]
    line_b = [b_p1_x,b_p1_y,b_p2_x,b_p2_y]
    return [ellipse_res,line_a,line_b]


#从检测的结果中保存为csv_格式
#input：detect_result 检测结果的列表
#      ：save_img_path 保存的原始图片的路径,其他文件名称路径从这个文件中解析

#列表的检测结果为：
#[org_img det_img mask_final box_result calcul_result ellipse_result,mask_index_range]
#原图  检测图 掩码图  检测框  计算长短轴  椭圆拟合结果 mask_index 索引的范围
#return:保存csv格式 以及对应的原图，检测图片，掩码图
def save_result(detect_result,save_img_path):
    org_img,det_img,mask_final_org,mask_final,box_result,calcul_result,ellipse_result,mask_index_range = detect_result


    det_out_file = save_img_path[:-4] + '_det.png'
    mask_out_org_file = save_img_path[:-4] + '_orgmask.png'
    mask_out_file = save_img_path[:-4] + '_mask.png'
    csv_out_file = save_img_path[:-4]+'.csv'


    cv2.imwrite(save_img_path, org_img)
    cv2.imwrite(det_out_file, det_img)
    cv2.imwrite(mask_out_file, mask_final)
    cv2.imwrite(mask_out_org_file, mask_final_org)

    f = open(csv_out_file,'w+')
    for i in range(len(box_result)):
        id = mask_index_range[0]+i;
        current_box = box_result[i];
        ellipse_res, line_a, line_b = ellipse_result[i]
        current_cal = calcul_result[i]
        # print(current_cal)
        #log info
        log_info = ("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n").format(id,
                                                    current_box[0],current_box[1],current_box[2],current_box[3],
                                                    current_cal[0],current_cal[1],
                                                    line_a[0], line_a[1],line_a[2], line_a[3],
                                                    line_b[0], line_b[1], line_b[2], line_b[3],
                                                    )
        # print(log_info)
        f.write(log_info)

    f.close()

def new_show_result_2(img, img_name, result, class_names, score_thr=0.3, mask_index = 5):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    org_img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    # draw segmentation masks
    bbox_color = (255, 128, 128)

    #读取深度图
    # 得到深度图的路径
    # depth_data_path = img[:-4] + '.raw'
    # depth_data = get_depth_data('./demo/5_Depth.raw')
    # current_depth_data = depth_data[310:330,170:190]
    #current_mean_dis = np.mean(current_depth_data)
    depth_data_path = img_name.split('/')[-1][:-4] + '.tif'
    # depth_data = None
    # if depth_data_path != 'none' and os.path.exists(depth_data_path):
    #     depth_data = cv2.imread(depth_data_path, -1)
    current_mean_dis = 2000
    if depth_data_path != 'none' and os.path.exists(depth_data_path):
        depth_data = cv2.imread(depth_data_path, -1)
        w, h = depth_data.shape
        current_depth_data = depth_data[w // 2 - 10:w // 2 + 10, h // 2 - 10:h // 2 + 10]
        current_mean_dis = np.mean(current_depth_data)

    #包含检测框和椭圆的结果
    detect_result = []
    box_result = []
    mask_result = []
    ellipse_result = []
    calcul_result = []
    num_result = []
    up_left_point_list = []
    mask_index_range = []



    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            ellipse_res,line_a,line_b = get_ellipse(mask)

            line_a_len = np.sqrt(np.square(line_a[2] - line_a[0]) + np.square(line_a[3] - line_a[1]))
            line_b_len = np.sqrt(np.square(line_b[2] - line_b[0]) + np.square(line_b[3] - line_b[1]))

            #
            # print('pic length is :%.2f,current dis is %.2f' %(line_a_len, current_mean_dis))

            # print('cal real length is :%.2f' %(cal_result))
            #
            # print('pic length is :%.2f,current dis is %.2f' % (line_b_len, current_mean_dis))

            # print('cal real length is :%.2f' % (cal_result))


            # cv2.ellipse(img,ellipse_res,(255,0,0),2)
            # cv2.line(img, (line_a[0],line_a[1]),(line_a[2],line_a[3]), (255, 0, 0), 2)
            # cv2.line(img, (line_b[0],line_b[1]),(line_b[2],line_b[3]), (255, 0, 0), 2)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
            bbox_int = bboxes[i].astype(np.int32)
            left_top = (bbox_int[0]+10, bbox_int[1]+10)
            right_bottom = (bbox_int[2]-10, bbox_int[3]-10)
            # cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)
            # current_mean_dis = 2000
            # if depth_data_path != 'none' and os.path.exists(depth_data_path):
            #     w, h = depth_data.shape
            #     current_depth_data = depth_data[bbox_int[0]:bbox_int[2], bbox_int[1]:bbox_int[3]]
            #     current_mean_dis = np.mean(current_depth_data)
            cal_result_a = line_a_len * current_mean_dis / (75 * 5)
            cal_result_b = line_b_len * current_mean_dis / (75 * 5)
            box_result.append([bbox_int[0],bbox_int[1],bbox_int[2],bbox_int[3]])
            up_left_point_list.append([bbox_int[1],bbox_int[0]])
            mask_result.append(mask)
            calcul_result.append([cal_result_b,cal_result_a])
            ellipse_result.append([ellipse_res,line_a,line_b])

            label = labels[i]
            # label_text = class_names[
            #     label] if class_names is not None else 'cls {}'.format(label)
            #label_text = ''
            # print(label_text)
            #cv2.putText(img, label_text, (int((bbox_int[0]+bbox_int[2])/2-5), int((bbox_int[1]+bbox_int[3])/2-1)),
                        #cv2.FONT_HERSHEY_COMPLEX,  0.25, (128, 0, 0))

        # label_text = 'num|{:d}'.format(len(inds))
        # cv2.putText(img, label_text, (20, 20),
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        num_result.append(len(inds))


    #对坐标卡框进行排序先上后下，先左后右

    sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
    #print(sort_index)
    # print(type(sort_index))
    #sort_index = np.array(sort_index,dtype=np.int)
    # sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
    box_result = [box_result[i] for i in sort_index]
    mask_result = [mask_result[i] for i in sort_index]
    calcul_result = [calcul_result[i] for i in sort_index]
    ellipse_result = [ellipse_result[i] for i in sort_index]

    mask_final = np.zeros((img.shape[0], img.shape[1],3))
    mask_final_org = np.zeros((img.shape[0], img.shape[1], 3))

    mask_index_val = int(mask_index)
    mask_index_range.append(mask_index_val)
    for i in range(len(box_result)):
        bbox_int = box_result[i]
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

        mask = mask_result[i]
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5

        result_n = []
        if(mask_index==5):
            result_n.append(mask_index_val)
            result_n.append(0)
            result_n.append(0)
        else:
            result_n.append(0)
            result_n.append(mask_index_val)
            result_n.append(0)
        # tmp = int(mask_index_val)
        # while(tmp!=0):
        #     n = tmp % 25
        #     result_n.append(n)
        #     tmp = int(tmp/25)
        # for i in range(len(result_n),3):
        #     result_n.append(0)
        # print(result_n)

        current_code_mask = np.stack([mask*result_n[j] for j in range(3)],axis=2)
        # print(current_code_mask.shape)
        mask_final_org = mask_final_org + current_code_mask

        if(bbox_int[2]-bbox_int[0]>50):
            iter_num = 6
        else:
            iter_num = 4
        kernel = np.ones(5, dtype=np.uint8)
        mask_er = cv2.erode(np.array(mask,dtype=np.uint8), kernel, iterations=iter_num)
        current_code_mask = np.stack([mask_er * result_n[j] for j in range(3)], axis=2)
        mask_final = mask_final + current_code_mask


        mask_index_val = mask_index_val + 2

        ellipse_res, line_a, line_b = ellipse_result[i]
        cv2.ellipse(img, ellipse_res, (255, 0, 0), 2)
        cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (255, 0, 0), 2)
        cv2.line(img, (line_b[0], line_b[1]), (line_b[2], line_b[3]), (255, 0, 0), 2)
        #cv2.imshow('img', img)

        #cv2.waitKey()
    return img, inds, calcul_result



def new_show_result_21(img, result, class_names, score_thr=0.3, mask_index = 5):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    org_img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    # draw segmentation masks
    bbox_color = (255, 128, 128)

    #读取深度图
    # 得到深度图的路径
    # depth_data_path = img[:-4] + '.raw'
    # depth_data = get_depth_data('./demo/5_Depth.raw')
    # current_depth_data = depth_data[310:330,170:190]
    #current_mean_dis = np.mean(current_depth_data)
    # depth_data_path = img.split('/')[-1][:-4] + '.tif'
    current_mean_dis = 2500
    # if depth_data_path != 'none' and os.path.exists(depth_data_path):
    #     depth_data = cv2.imread(depth_data_path, -1)
    #     w, h = depth_data.shape
    #     current_depth_data = depth_data[w // 2 - 10:w // 2 + 10, h // 2 - 10:h // 2 + 10]
    #     current_mean_dis = np.mean(current_depth_data)

    #包含检测框和椭圆的结果
    detect_result = []
    box_result = []
    mask_result = []
    ellipse_result = []
    calcul_result = []
    num_result = []
    up_left_point_list = []
    mask_index_range = []



    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            ellipse_res,line_a,line_b = get_ellipse(mask)

            line_a_len = np.sqrt(np.square(line_a[2] - line_a[0]) + np.square(line_a[3] - line_a[1]))
            line_b_len = np.sqrt(np.square(line_b[2] - line_b[0]) + np.square(line_b[3] - line_b[1]))
            cal_result_a = line_a_len * current_mean_dis / (75 * 5)
            cal_result_b = line_b_len * current_mean_dis / (75 * 5)
            #
            # print('pic length is :%.2f,current dis is %.2f' %(line_a_len, current_mean_dis))

            # print('cal real length is :%.2f' %(cal_result))
            #
            # print('pic length is :%.2f,current dis is %.2f' % (line_b_len, current_mean_dis))

            # print('cal real length is :%.2f' % (cal_result))


            # cv2.ellipse(img,ellipse_res,(255,0,0),2)
            # cv2.line(img, (line_a[0],line_a[1]),(line_a[2],line_a[3]), (255, 0, 0), 2)
            # cv2.line(img, (line_b[0],line_b[1]),(line_b[2],line_b[3]), (255, 0, 0), 2)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
            bbox_int = bboxes[i].astype(np.int32)
            left_top = (bbox_int[0]+10, bbox_int[1]+10)
            right_bottom = (bbox_int[2]-10, bbox_int[3]-10)
            # cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

            box_result.append([bbox_int[0],bbox_int[1],bbox_int[2],bbox_int[3]])
            up_left_point_list.append([bbox_int[1],bbox_int[0]])
            mask_result.append(mask)
            calcul_result.append([cal_result_b,cal_result_a])
            ellipse_result.append([ellipse_res,line_a,line_b])

            label = labels[i]
            # label_text = class_names[
            #     label] if class_names is not None else 'cls {}'.format(label)
            #label_text = ''
            # print(label_text)
            #cv2.putText(img, label_text, (int((bbox_int[0]+bbox_int[2])/2-5), int((bbox_int[1]+bbox_int[3])/2-1)),
                        #cv2.FONT_HERSHEY_COMPLEX,  0.25, (128, 0, 0))

        # label_text = 'num|{:d}'.format(len(inds))
        # cv2.putText(img, label_text, (20, 20),
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        num_result.append(len(inds))


    #对坐标卡框进行排序先上后下，先左后右

    sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
    #print(sort_index)
    # print(type(sort_index))
    #sort_index = np.array(sort_index,dtype=np.int)
    # sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
    box_result = [box_result[i] for i in sort_index]
    mask_result = [mask_result[i] for i in sort_index]
    calcul_result = [calcul_result[i] for i in sort_index]
    ellipse_result = [ellipse_result[i] for i in sort_index]

    mask_final = np.zeros((img.shape[0], img.shape[1],3))
    mask_final_org = np.zeros((img.shape[0], img.shape[1], 3))

    mask_index_val = int(mask_index)
    mask_index_range.append(mask_index_val)
    for i in range(len(box_result)):
        bbox_int = box_result[i]
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

        mask = mask_result[i]
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5

        result_n = []
        if(mask_index==5):
            result_n.append(mask_index_val)
            result_n.append(0)
            result_n.append(0)
        else:
            result_n.append(0)
            result_n.append(mask_index_val)
            result_n.append(0)
        # tmp = int(mask_index_val)
        # while(tmp!=0):
        #     n = tmp % 25
        #     result_n.append(n)
        #     tmp = int(tmp/25)
        # for i in range(len(result_n),3):
        #     result_n.append(0)
        # print(result_n)

        current_code_mask = np.stack([mask*result_n[j] for j in range(3)],axis=2)
        # print(current_code_mask.shape)
        mask_final_org = mask_final_org + current_code_mask

        if(bbox_int[2]-bbox_int[0]>50):
            iter_num = 6
        else:
            iter_num = 4
        kernel = np.ones(5, dtype=np.uint8)
        mask_er = cv2.erode(np.array(mask,dtype=np.uint8), kernel, iterations=iter_num)
        current_code_mask = np.stack([mask_er * result_n[j] for j in range(3)], axis=2)
        mask_final = mask_final + current_code_mask


        mask_index_val = mask_index_val + 2

        ellipse_res, line_a, line_b = ellipse_result[i]
        cv2.ellipse(img, ellipse_res, (255, 0, 0), 2)
        cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (255, 0, 0), 2)
        cv2.line(img, (line_b[0], line_b[1]), (line_b[2], line_b[3]), (255, 0, 0), 2)
        #cv2.imshow('img', img)

        #cv2.waitKey()
    return img, inds, calcul_result

# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None,mask_index=1):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    org_img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    # draw segmentation masks
    bbox_color = (255, 128, 128)

    #读取深度图
    # 得到深度图的路径
    # depth_data_path = img[:-4] + '.raw'
    # depth_data = get_depth_data('./demo/5_Depth.raw')
    # current_depth_data = depth_data[310:330,170:190]
    #current_mean_dis = np.mean(current_depth_data)
    current_mean_dis = 0

    #包含检测框和椭圆的结果
    detect_result = []
    box_result = []
    mask_result = []
    ellipse_result = []
    calcul_result = []
    num_result = []
    up_left_point_list = []
    mask_index_range = []



    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            ellipse_res,line_a,line_b = get_ellipse(mask)

            line_a_len = np.sqrt(np.square(line_a[2] - line_a[0]) + np.square(line_a[3] - line_a[1]))
            line_b_len = np.sqrt(np.square(line_b[2] - line_b[0]) + np.square(line_b[3] - line_b[1]))
            cal_result_a = line_a_len * current_mean_dis / (75 * 5)
            cal_result_b = line_b_len * current_mean_dis / (75 * 5)
            #
            # print('pic length is :%.2f,current dis is %.2f' %(line_a_len, current_mean_dis))

            # print('cal real length is :%.2f' %(cal_result))
            #
            # print('pic length is :%.2f,current dis is %.2f' % (line_b_len, current_mean_dis))

            # print('cal real length is :%.2f' % (cal_result))


            # cv2.ellipse(img,ellipse_res,(255,0,0),2)
            # cv2.line(img, (line_a[0],line_a[1]),(line_a[2],line_a[3]), (255, 0, 0), 2)
            # cv2.line(img, (line_b[0],line_b[1]),(line_b[2],line_b[3]), (255, 0, 0), 2)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
            bbox_int = bboxes[i].astype(np.int32)
            left_top = (bbox_int[0]+10, bbox_int[1]+10)
            right_bottom = (bbox_int[2]-10, bbox_int[3]-10)
            # cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

            box_result.append([bbox_int[0],bbox_int[1],bbox_int[2],bbox_int[3]])
            up_left_point_list.append([bbox_int[1],bbox_int[0]])
            mask_result.append(mask)
            calcul_result.append([cal_result_b,cal_result_a])
            ellipse_result.append([ellipse_res,line_a,line_b])

            label = labels[i]
            # label_text = class_names[
            #     label] if class_names is not None else 'cls {}'.format(label)
            #label_text = ''
            # print(label_text)
            #cv2.putText(img, label_text, (int((bbox_int[0]+bbox_int[2])/2-5), int((bbox_int[1]+bbox_int[3])/2-1)),
                        #cv2.FONT_HERSHEY_COMPLEX,  0.25, (128, 0, 0))

        # label_text = 'num|{:d}'.format(len(inds))
        # cv2.putText(img, label_text, (20, 20),
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        num_result.append(len(inds))


    #对坐标卡框进行排序先上后下，先左后右

    sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
    #print(sort_index)
    # print(type(sort_index))
    #sort_index = np.array(sort_index,dtype=np.int)
    # sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
    box_result = [box_result[i] for i in sort_index]
    mask_result = [mask_result[i] for i in sort_index]
    calcul_result = [calcul_result[i] for i in sort_index]
    ellipse_result = [ellipse_result[i] for i in sort_index]

    mask_final = np.zeros((img.shape[0], img.shape[1],3))
    mask_final_org = np.zeros((img.shape[0], img.shape[1], 3))

    mask_index_val = int(mask_index)
    mask_index_range.append(mask_index_val)
    for i in range(len(box_result)):
        bbox_int = box_result[i]
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

        mask = mask_result[i]
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5

        result_n = []
        if(mask_index==5):
            result_n.append(mask_index_val)
            result_n.append(0)
            result_n.append(0)
        else:
            result_n.append(0)
            result_n.append(mask_index_val)
            result_n.append(0)
        # tmp = int(mask_index_val)
        # while(tmp!=0):
        #     n = tmp % 25
        #     result_n.append(n)
        #     tmp = int(tmp/25)
        # for i in range(len(result_n),3):
        #     result_n.append(0)
        # print(result_n)

        current_code_mask = np.stack([mask*result_n[j] for j in range(3)],axis=2)
        # print(current_code_mask.shape)
        mask_final_org = mask_final_org + current_code_mask

        if(bbox_int[2]-bbox_int[0]>50):
            iter_num = 6
        else:
            iter_num = 4
        kernel = np.ones(5, dtype=np.uint8)
        mask_er = cv2.erode(np.array(mask,dtype=np.uint8), kernel, iterations=iter_num)
        current_code_mask = np.stack([mask_er * result_n[j] for j in range(3)], axis=2)
        mask_final = mask_final + current_code_mask


        mask_index_val = mask_index_val + 2

        ellipse_res, line_a, line_b = ellipse_result[i]
        cv2.ellipse(img, ellipse_res, (255, 0, 0), 2)
        cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (255, 0, 0), 2)
        cv2.line(img, (line_b[0], line_b[1]), (line_b[2], line_b[3]), (255, 0, 0), 2)
        #cv2.imshow('img', img)

        #cv2.waitKey()
    mask_index_range.append(mask_index_val)
    mask_final = np.uint8(mask_final)

    detect_result = [org_img,img,mask_final_org,mask_final,box_result,calcul_result,ellipse_result,mask_index_range]

    save_result(detect_result,out_file)
    # mmcv.imshow_det_bboxes

    # img = Image.fromarray(np.uint8(img))
    # img.show()
    #cv2.imshow('org',img)
    # mask_final = np.uint8(mask_final)
    #cv2.imshow('mask_final',mask_final)


    #cv2.waitKey()
    # img.save(out_file)
    # det_out_file = out_file[:-4] + '_det.png'
    # cv2.imwrite(det_out_file,img)
    # mask_out_file = out_file[:-4]+'_mask.png'
    # cv2.imwrite(mask_out_file, mask_final)
    dir(img)
    return mask_index_val+1

    #这里返回下个图的index索引
    # img.close()
    # # draw bounding boxes
    # labels = [
    #     np.full(bbox.shape[0], i, dtype=np.int32)
    #     for i, bbox in enumerate(bbox_result)
    # ]
    # labels = np.concatenate(labels)
    # mmcv.imshow_det_bboxes(
    #     img.copy(),
    #     bboxes,
    #     labels,
    #     class_names=class_names,
    #     score_thr=score_thr,
    #     show=out_file is None,
    #     out_file=out_file)



# TODO: merge this method with the one in BaseDetector
def new_show_result_3(img, result, class_names, img_path, score_thr=0.3, out_file=None,mask_index=0):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    org_img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    # draw segmentation masks
    bbox_color = (255, 128, 128)

    #读取深度图
    # 得到深度图的路径
    # depth_data_path = img[:-4] + '.raw'
    depth_data_path = img_path
    current_mean_dis = 2000
    if depth_data_path != 'none' and os.path.exists(depth_data_path):
        depth_data = cv2.imread(depth_data_path, -1)
        w, h = depth_data.shape
        current_depth_data = depth_data[w // 2 - 10:w // 2 + 10, h // 2 - 10:h // 2 + 10]
        current_mean_dis = np.mean(current_depth_data)
    #current_mean_dis = 0
    # print('current_mean_dis:', current_mean_dis)
    #包含检测框和椭圆的结果
    detect_result = []
    box_result = []
    mask_result = []
    ellipse_result = []
    calcul_result = []
    num_result = []
    up_left_point_list = []
    mask_index_range = []



    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)

            #
            # print('pic length is :%.2f,current dis is %.2f' %(line_a_len, current_mean_dis))

            # print('cal real length is :%.2f' %(cal_result))
            #
            # print('pic length is :%.2f,current dis is %.2f' % (line_b_len, current_mean_dis))

            # print('cal real length is :%.2f' % (cal_result))


            # cv2.ellipse(img,ellipse_res,(255,0,0),2)
            # cv2.line(img, (line_a[0],line_a[1]),(line_a[2],line_a[3]), (255, 0, 0), 2)
            # cv2.line(img, (line_b[0],line_b[1]),(line_b[2],line_b[3]), (255, 0, 0), 2)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
            bbox_int = bboxes[i].astype(np.int32)
            left_top = (bbox_int[0]+10, bbox_int[1]+10)
            right_bottom = (bbox_int[2]-10, bbox_int[3]-10)

            if (np.abs(bbox_int[2] - bbox_int[0]) < 30 or np.abs(bbox_int[3] - bbox_int[1]) < 30):
                continue
            ellipse_res, line_a, line_b = get_ellipse(mask)

            line_a_len = np.sqrt(np.square(line_a[2] - line_a[0]) + np.square(line_a[3] - line_a[1]))
            line_b_len = np.sqrt(np.square(line_b[2] - line_b[0]) + np.square(line_b[3] - line_b[1]))
            cal_result_a = line_a_len * current_mean_dis / (75 * 5)
            cal_result_b = line_b_len * current_mean_dis / (75 * 5)
            # cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

            box_result.append([bbox_int[0],bbox_int[1],bbox_int[2],bbox_int[3]])
            up_left_point_list.append([bbox_int[1],bbox_int[0]])
            mask_result.append(mask)
            calcul_result.append([cal_result_b,cal_result_a])
            ellipse_result.append([ellipse_res,line_a,line_b])

            label = labels[i]
            # label_text = class_names[
            #     label] if class_names is not None else 'cls {}'.format(label)
            #label_text = ''
            # print(label_text)
            #cv2.putText(img, label_text, (int((bbox_int[0]+bbox_int[2])/2-5), int((bbox_int[1]+bbox_int[3])/2-1)),
                        #cv2.FONT_HERSHEY_COMPLEX,  0.25, (128, 0, 0))

        # label_text = 'num|{:d}'.format(len(inds))
        # cv2.putText(img, label_text, (20, 20),
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        num_result.append(len(inds))


    #对坐标卡框进行排序先上后下，先左后右

    sort_index = sorted(range(len(up_left_point_list)),key=lambda x:(up_left_point_list[x][0],up_left_point_list[x][1]))
    # print(sort_index)
    # print(type(sort_index))
    #sort_index = np.array(sort_index,dtype=np.int)
    # sort_index=up_left_point_list.argsort(key=lambda x:(x[0],x[1]))
    box_result = [box_result[i] for i in sort_index]
    mask_result = [mask_result[i] for i in sort_index]
    calcul_result = [calcul_result[i] for i in sort_index]
    ellipse_result = [ellipse_result[i] for i in sort_index]

    mask_final = np.zeros((img.shape[0], img.shape[1],3))
    mask_final_org = np.zeros((img.shape[0], img.shape[1], 3))

    mask_index_val = 50
    mask_index_range.append(mask_index_val)
    for i in range(len(box_result)):
        bbox_int = box_result[i]
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])


        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=1)

        mask = mask_result[i]
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5

        result_n = []
        if(mask_index==0):
            result_n.append(mask_index_val)
            result_n.append(0)
            result_n.append(0)
        elif (mask_index == 1):
            result_n.append(0)
            result_n.append(mask_index_val)
            result_n.append(0)
        else:
            result_n.append(0)
            result_n.append(0)
            result_n.append(mask_index_val)
        # tmp = int(mask_index_val)
        # while(tmp!=0):
        #     n = tmp % 25
        #     result_n.append(n)
        #     tmp = int(tmp/25)
        # for i in range(len(result_n),3):
        #     result_n.append(0)
        # print(result_n)

        current_code_mask = np.stack([mask*result_n[j] for j in range(3)],axis=2)
        # print(current_code_mask.shape)
        mask_final_org = mask_final_org + current_code_mask

        if(bbox_int[2]-bbox_int[0]>60 or bbox_int[3]-bbox_int[1]>60 ):
            iter_num = 11
        else:
            iter_num = 6
        kernel = np.ones(5, dtype=np.uint8)
        mask_er = cv2.erode(np.array(mask,dtype=np.uint8), kernel, iterations=iter_num)
        current_code_mask = np.stack([mask_er * result_n[j] for j in range(3)], axis=2)
        mask_final = mask_final + current_code_mask


        mask_index_val = mask_index_val + 2

        ellipse_res, line_a, line_b = ellipse_result[i]
        cv2.ellipse(img, ellipse_res, (255, 0, 0), 2)
        cv2.line(img, (line_a[0], line_a[1]), (line_a[2], line_a[3]), (255, 0, 0), 2)
        cv2.line(img, (line_b[0], line_b[1]), (line_b[2], line_b[3]), (255, 0, 0), 2)
        # cv2.imshow('img', img)
        #
        # cv2.waitKey()
    mask_index_range.append(mask_index_val)
    mask_final = np.uint8(mask_final)

    detect_result = [org_img,img,mask_final_org,mask_final,box_result,calcul_result,ellipse_result,mask_index_range]

    save_result(detect_result,out_file)
    # mmcv.imshow_det_bboxes
    # img.save(out_file)
    # det_out_file = out_file[:-4] + '_det.png'
    # cv2.imwrite(det_out_file,img)
    # mask_out_file = out_file[:-4]+'_mask.png'
    # cv2.imwrite(mask_out_file, mask_final)
    return mask_index_val+1


#这个函数用来对应的匹配文件
txt_info='''{{center_image_index | {0} | center image index}}
{{center_image_rotation_angle | {1} | center image rotation angle}}
{{images_count | {2} | images count}}
{3}'''
def creat_match_graph_txt(img_num=5,root_path=None, root_path_mask=None, file_name=None):
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
            one_item_info = one_item_info_tm.format(i,str(i-1)+','+str(i+1),i)
        all_item_info = all_item_info + one_item_info

    final_txt = txt_info.format(int(img_num/2),0,img_num,all_item_info)
    # print(final_txt)
    with open(root_path + '/' + file_name + '.txt', 'w') as f:
        f.write(final_txt)
    with open(root_path_mask + '/' + file_name + '.txt', 'w') as f:
        f.write(final_txt)
    ###############
    #
    #需要保存到对应的可以执行拼接程序的目录中 ，需要注意名字格式
    #
    ###############

def get_result_from_meger_file(file_path,img_path):
    cal_list = []
    img = cv2.imread(img_path)
    f = open(file_path,'r')
    all_lines = f.readlines()
    f.close()

    count_num = len(all_lines)

    for i in range(len(all_lines)):
        pass
        line = all_lines[i]
        line_s = line.split(',')
        x, y, x2, y2 = int(line_s[-4]), int(line_s[-3]), int(line_s[-2]), int(line_s[-1])
        cal1,cal2 = float(line_s[5]),float(line_s[6])
        label_text = str(i)
        cv2.putText(img, label_text,(x+10,y+10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0))
        cal_list.append([cal1,cal2])
    cv2.imwrite('results/out_notes.png', img)
    #cv2.imshow('text',img)
    #print("all_detection_num is ",count_num,len(cal_list))
    #print(cal_list)
    #cv2.waitKey()
    return  img, cal_list, count_num