from torch.nn import L1Loss, MSELoss, HuberLoss
import torch
import numpy as np
import torch
import torch.nn as nn
from datasets import A2D2_box
# from utils import intersection_over_union
import glob

import time
from torchvision.ops import box_iou
import math
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=14, num_boxes=10, lambda_coord=5.0, lambda_noobj=5.0,lambda_loc=7.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_loc=lambda_loc


        self.loss_function=nn.MSELoss()

    def forward(self, predictions, targets):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted

        target_confidence = targets[0:,0: self.num_boxes*19+0:19]
        target_boxes_x = targets    [0:,1: self.num_boxes*19+1:19]
        target_boxes_y = targets    [0:,2: self.num_boxes*19+2:19]
        target_width = targets      [0:,3: self.num_boxes*19+3:19]
        target_height = targets     [0:,4: self.num_boxes*19+4:19]

        target_exist=torch.nonzero(target_confidence)

        valid_grids = torch.unique(torch.stack((target_exist[:, 0], target_exist[:, 1], target_exist[:, 2],target_exist[:, 3]), dim=1), dim=0)



        pred_confidence = predictions[0:,0: self.num_boxes*19+0:19]
        pred_boxes_x = predictions   [0:,1: self.num_boxes*19+1:19]
        pred_boxes_y = predictions   [0:,2: self.num_boxes*19+2:19]
        pred_width = predictions     [0:,3: self.num_boxes*19+3:19]
        pred_height = predictions    [0:,4: self.num_boxes*19+4:19]



        iou=self.IOUcalc(pred_boxes_x,pred_boxes_y,pred_width,pred_height,
            target_boxes_x,target_boxes_y,target_width,target_height,valid_grids)
        
   

        localization_loss=0.0
        confidence_loss_with_object=0.0
        class_loss=0.0
        confidence_loss_nothing=(target_confidence-pred_confidence)**2


        for grids in valid_grids:

            x_coord_loss=self.loss_function(target_boxes_x[grids[0]][grids[1]][grids[2]][grids[3]] ,  pred_boxes_x[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[2]]][grids[2]][grids[3]] )
            y_coord_loss=self.loss_function(target_boxes_y[grids[0]][grids[1]][grids[2]][grids[3]] ,  pred_boxes_y[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[2]]][grids[2]][grids[3]] )

            width_loss  = (torch.sqrt(torch.clamp(target_width [grids[0]][grids[1]][grids[2]][grids[3]], min=0.0)) - torch.sqrt(torch.clamp(pred_width [grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[3]]][grids[2]][grids[3]], min=0.0)))**2
            height_loss = (torch.sqrt(torch.clamp(target_height[grids[0]][grids[1]][grids[2]][grids[3]], min=0.0)) - torch.sqrt(torch.clamp(pred_height[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[3]]][grids[2]][grids[3]], min=0.0)))**2



            if math.isnan(x_coord_loss):
                print('guilty is x', predictions)
                exit()
            if math.isnan(y_coord_loss):
                print('guilty is y')
                exit()
            if math.isnan(width_loss):
                print('guilty is width')
                exit()
            if math.isnan(height_loss):
                print('guilty is height')
                exit()

            # x_coord_loss=x_coord_loss*1.8
            # y_coord_loss=y_coord_loss*1.8

            localization_loss=localization_loss+x_coord_loss+y_coord_loss+width_loss+height_loss

            confidence_loss_with_object=confidence_loss_with_object+(target_confidence[grids[0]][grids[1]][grids[2]][grids[3]]-pred_confidence[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[3]]][grids[2]][grids[3]])**2
            confidence_loss_nothing[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[3]]][grids[2]][grids[3]]=0.0

            pred_class=predictions[grids[0]][iou[grids[0]][grids[1]][grids[2]][grids[3]]*19+5:iou[grids[0]][grids[1]][grids[2]][grids[3]]*19+5+self.num_classes] [grids[2]][grids[3]]
            target_class=targets[grids[0]][grids[0]*19+5:grids[0]*19+5+self.num_classes] [grids[2]][grids[3]] 

            class_loss=class_loss+self.loss_function(pred_class,target_class)




        localization_loss=localization_loss/len(valid_grids)
        confidence_loss_with_object=confidence_loss_with_object/(len(valid_grids))
        class_loss=class_loss/len(valid_grids)


        confidence_loss_nothing=confidence_loss_nothing.mean()

        





        confidence_loss_nothing=confidence_loss_nothing*self.lambda_noobj
        confidence_loss_with_object=confidence_loss_with_object*self.lambda_coord
        localization_loss=localization_loss*self.lambda_loc



        total_loss=localization_loss+confidence_loss_with_object+confidence_loss_nothing+class_loss





        return total_loss
    

    def IOUcalc(self,pred_boxes_x,pred_boxes_y,pred_width,pred_height,
                target_boxes_x,target_boxes_y,target_width,target_height,valid_grids):
        pred_boxes_x_1=pred_boxes_x-(pred_width/2)
        pred_boxes_x_2=pred_boxes_x+(pred_width/2)
    
        pred_boxes_y_1=pred_boxes_y-(pred_height/2)
        pred_boxes_y_2=pred_boxes_y+(pred_height/2)
    
        target_boxes_x_1=target_boxes_x-(target_width/2)
        target_boxes_x_2=target_boxes_x+(target_width/2)
    
        target_boxes_y_1=target_boxes_y-(target_height/2)
        target_boxes_y_2=target_boxes_y+(target_height/2)
    
        pred_boxes=torch.stack((pred_boxes_x_1,pred_boxes_y_1,pred_boxes_x_2,pred_boxes_y_2),axis=-1)
        target_boxes=torch.stack((target_boxes_x_1,target_boxes_y_1,target_boxes_x_2,target_boxes_y_2),axis=-1)
    
    
        # pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
        # target_boxes = torch.tensor(target_boxes, dtype=torch.float32)
    
    
        # print(pred_boxes.shape)
        # print(np.max(valid_grids[:,2]))

        # exit()
        max_IOU=torch.zeros((target_boxes.shape[0],target_boxes.shape[1],target_boxes.shape[2],target_boxes.shape[3]),dtype=int)
        # max_IOU=np.zeros((valid_grids.shape[0],valid_grids.shape[1],valid_grids.shape[2],valid_grids.shape[3]))

        # print(target_boxes.shape)
        # exit()

        for grids in valid_grids:
            for anchor_box in range(self.num_boxes):
                # print(grids)
                # print(pred_boxes[anchor_box][grids[0]][grids[1]])
                target_box=target_boxes[grids[0]][grids[1]][grids[2]][grids[3]]
                pred_box=pred_boxes[grids[0]][anchor_box][grids[2]][grids[3]]
                iou=intersection_over_union(target_box,pred_box,'corners')
                if max_IOU[grids[0]][grids[1]][grids[2]][grids[3]] < iou and anchor_box not in max_IOU[grids[0]][:][grids[2]][grids[3]]:
                    max_IOU[grids[0]][grids[1]][grids[2]][grids[3]]=anchor_box

        


        return max_IOU
    

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "corners":
        # box1_x1 = boxes_preds[..., 0:1]
        # box1_y1 = boxes_preds[..., 1:2]
        # box1_x2 = boxes_preds[..., 2:3]
        # box1_y2 = boxes_preds[..., 3:4]
        # box2_x1 = boxes_labels[..., 0:1]
        # box2_y1 = boxes_labels[..., 1:2]
        # box2_x2 = boxes_labels[..., 2:3]
        # box2_y2 = boxes_labels[..., 3:4]
        box1_x1 = boxes_preds[0]
        box1_y1 = boxes_preds[1]
        box1_x2 = boxes_preds[2]
        box1_y2 = boxes_preds[3]  # (N, 1)
        box2_x1 = boxes_labels[0]
        box2_y1 = boxes_labels[1]
        box2_x2 = boxes_labels[2]
        box2_y2 = boxes_labels[3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    result=intersection / (box1_area + box2_area - intersection + 1e-6)

    return result


class ScaleInvariantLoss(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self,predicted_depths, goal_depths, boundaries ):
        # predicted_depths, goal_depths, boundaries = 
        depth_ratio_map = torch.log(boundaries * predicted_depths + self.epsilon) - \
                          torch.log(boundaries * goal_depths + self.epsilon)

        weighted_sum = torch.sum(boundaries, dim=(1, 2, 3))
        # print(depth_ratio_map)
        # print(weighted_sum)
        # exit()
        loss_1 = torch.sum(depth_ratio_map * depth_ratio_map,
                           dim=(1, 2, 3)) / weighted_sum
        sum_2 = torch.sum(depth_ratio_map, dim=(1, 2, 3))
        loss_2 = (sum_2 * sum_2) / (weighted_sum * weighted_sum)
        return torch.mean(loss_1 + loss_2)



class DepthLoss(nn.Module):
    def __init__(self,loss_function=ScaleInvariantLoss()):
    # def __init__(self,loss_function=nn.BCEWithLogitsLoss()):
        super(DepthLoss, self).__init__()

        self.loss=loss_function

    def forward(self, predictions, targets):

        non_zero_mask = targets != 0

        # Masked predictions and targets
        masked_predictions = predictions[non_zero_mask]
        masked_targets = targets[non_zero_mask]
        # non_zero_mask=non_zero_mask.int()
        # print(non_zero_mask)


        # Calculate loss
        loss = self.loss(predictions, targets,non_zero_mask)
        # loss=self.loss(masked_predictions,masked_targets)

        return loss

# import torch
# import torch.nn as nn
# from torch.autograd import Variable

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
# depth_loss = DepthLoss()
# # Create some sample predictions and targets tensors
# predictions = torch.rand(16, 1, 256, 256)
# targets = torch.rand(16, 1, 256, 256)
# # Calculate the loss
# loss = depth_loss(predictions, targets)
# print("RMSE Loss:", loss.item())
# predictions = torch.randn(4,44, 1024, 1024)
# targets = torch.rand(4,44, 1024, 1024)

# criterion = nn.CrossEntropyLoss()

# start_time = time.time()

# loss = criterion(predictions, targets)
# end_time = time.time()

# print('Segmentation Loss Time: ', end_time - start_time)

# batch_size = 4
# S = 8  # Grid size
# C = 14  # Number of classes
# B = 10   # Number of bounding boxes
# files=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/2018*/camera/cam_front_center/*.png"))


# # Dummy predictions and targets
# predictions = torch.randn(batch_size, B * (5 + C),S,S)
# # targets = torch.randn(batch_size, S, S, B, 5 + C)
# # print(files[0])
# target = A2D2_box(files)

# from torch.utils.data import DataLoader
# dataloader = DataLoader(target, batch_size=batch_size)
# # targets=target[0:3]['A2D2_box']

# # targets=targets.unsqueeze(0)
# # Instantiate the custom YOLO loss



# custom_loss = YOLOLoss(num_classes=C, num_boxes=B)
# for batch in dataloader:
#     inputs, targets = batch['image'], batch['A2D2_box']

#     start_time = time.time()

#     loss = custom_loss(predictions, targets)
#     end_time = time.time()

#     print('Yolo Loss Time: ', end_time - start_time)

# # # Print the loss
#     print("Custom YOLO Loss:", loss.item())
#     exit()