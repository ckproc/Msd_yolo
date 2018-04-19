
import numpy as np


def py_cpu_tensor_to_box(det_list, num_box, im_size, pool):

    assert len(det_list) == 4
    
    output_row = det_list[0].shape[1]
    output_col = det_list[0].shape[2]
    num_class = (det_list[0].shape[3] / (2*num_box)) - 3 - output_row - output_col
    
    param = []
    for k in xrange(4):
        det = det_list[k][0]
        for i in xrange(output_row):
            for j in xrange(output_col):
                param.append([det, k, i, j, num_box, num_class, output_row, output_col, im_size])

    boxes_list = pool.map(func, param)
    boxes = [[] for _ in xrange(num_class)]
    for i in xrange(num_class):
        for j in xrange(len(boxes_list)):
            boxes[i] += boxes_list[j][i]

    boxes = [np.float32(x) for x in boxes]
    return boxes


def func(item):
    det, k, i, j, num_box, num_class, output_row, output_col, im_size = item
    ss = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
    boxes = [[] for _ in xrange(num_class)]
    p_start, p_end = get_range(i, output_row, ss[k][1])
    q_start, q_end = get_range(j, output_col, ss[k][0])
    for p in xrange(p_start, p_end):
        for q in xrange(q_start, q_end):
            for n in xrange(num_box):

                pm = det[i, j, n]
                pc = det[p, q, num_box + n]
                pconnm = det[i, j, num_box*6 \
                            + n*(output_row+output_col) + p] \
                       * det[i, j, num_box*6 \
                            + n*(output_row+output_col)+output_row + q]
                pconnc = det[p, q, num_box*6 \
                            + (num_box+n)*(output_row+output_col) + i] \
                       * det[p, q, num_box*6 \
                            + (num_box+n)*(output_row+output_col)+output_row + j]
                score_part = pm * pc * (pconnm + pconnc) / 2.0
                if score_part < 0.001:
                  continue

                xm = (det[i, j, (num_box+n)*2 + 0] + j) / output_col * im_size[0]
                ym = (det[i, j, (num_box+n)*2 + 1] + i) / output_row * im_size[1]
                xc = (det[p, q, (num_box*2+n)*2 + 0] + q) / output_col * im_size[0]
                yc = (det[p, q, (num_box*2+n)*2 + 1] + p) / output_row * im_size[1]

                w_2 = ss[k][0] * (xc - xm)
                h_2 = ss[k][1] * (yc - ym)
                x1 = xm - w_2; y1 = ym - h_2
                x2 = xm + w_2; y2 = ym + h_2

                for cls in xrange(num_class):
                    pclsm = det[i, j, num_box*(6+2*(output_row+output_col)) \
                            + n*num_class + cls]
                    pclsc = det[p, q, num_box*(6+2*(output_row+output_col)+num_class) \
                            + n*num_class + cls]

                    score = score_part * pclsm * pclsc
                    if score < 0.001:
                        continue
                    boxes[cls].append([x1, y1, x2, y2, score])
    return boxes

    
def get_range(i, rows, ss):
    if i < rows/2:
        if ss < 0:
            return 0, i + 1
        else:
            return i, 2*(i + 1)
    else:
        if ss < 0:
            return 2*i - rows, i + 1
        else:
            return i, rows
