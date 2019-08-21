from goods.freezer.keras_yolo3.util import iou_util


#没有交叠的目标商品 阈值应该比较高（主要解决冰柜中，孤立商品多检的问题）
def  single_filter(iou,min_score,classes,scores,boxes):
    single_flag = []
    for i in range(len(classes)):
        flg = True
        for j in range(len(classes)):
            if i != j:
                if iou_util.IOU(boxes[i], boxes[j])>iou:
                    flg=False
        if flg and scores[i] <= min_score:
            single_flag.append(True)
        else:
            single_flag.append(False)
    p_classes, p_scores, p_boxes = [],[],[]
    for fg,cls,sce,bx in zip(single_flag,classes,scores,boxes):
        if fg == False:
            p_classes.append(cls)
            p_scores.append(sce)
            p_boxes.append(bx)
    return p_classes, p_scores, p_boxes


#存在交叠的目标商品，选择阈值较高的那个（yolov3中，对交叠的同一类目标做了最大IOU限制，而不同类的目标没有做处理，造成一定多检现象）
def diff_fiter(iou,classes,scores,boxes):
    diff_flag = []
    for i in range(len(classes)):
        flag = False
        for j in range(len(classes)):
            if classes[i] != classes[j]:
                if iou_util.IOU(boxes[i], boxes[j]) > iou:
                    if scores[i] < scores[j]:
                        flag = True
                        break
        diff_flag.append(flag)
    p_classes, p_scores, p_boxes = [], [], []
    for fg, cls, sce, bx in zip(diff_flag, classes, scores, boxes):
        if fg == False:
            p_classes.append(cls)
            p_scores.append(sce)
            p_boxes.append(bx)
    return p_classes, p_scores, p_boxes