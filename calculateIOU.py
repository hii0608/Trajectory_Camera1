import pandas as pd


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

if __name__ == '__main__':
#   df  = pd.read_csv('./camera1/result/xyMSE_RMSE-testNum1173-horison01-20.csv')
  df  = pd.read_csv('./camera1/result/xyMSE_RMSE-testNum1173-horizon_01-20noscaler.csv')
  xRMSE = df['xlossRMSE']
  yRMSE = df['ylossRMSE']
  iouList = []
  for i in range(len(xRMSE)) : 
    box1 = (0,0,181,199)
    box2 = (xRMSE[i]+0, yRMSE[i]+0,xRMSE[i]+181, yRMSE[i]+199 )
    iou = IoU(box1, box2)
    iouList.append(iou)

df = pd.DataFrame(
    {"horizon": list(range(1,21)) , "IoU" : iouList
    })
df.to_csv('./camera1/result/iou01.csv')
print(df)



  


 
