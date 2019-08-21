
def IOU (predictBox,Box):
    cx1 = predictBox[0]
    cy1 = predictBox[1]
    cx2 = predictBox[2]
    cy2 = predictBox[3]
    gx1 = Box[0]
    gy1 = Box[1]
    gx2 = Box[2]
    gy2 = Box[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = float(area) / (carea + garea - area)
    return iou