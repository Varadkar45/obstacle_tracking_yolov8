import hydra
import torch
import cv2
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

tracker = None

def init_tracker():
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker =Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)

rand_color_list = []
    
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        
        
    return img

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
        


class DetectionPredictor(BasePredictor):
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        # tracker
        self.data_path = p
    
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    
    
        # #..................USE TRACK FUNCTION....................
        dets_to_sort = np.empty((0,6))
        
        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x1, y1, x2, y2, conf, detclass])))
        
        tracked_dets = tracker.update(dets_to_sort)
        tracks =tracker.getTrackers()
        
        for track in tracks:
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        rand_color_list[track.id], thickness=3) 
                        for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 
        

        if len(tracked_dets)>0:
            bbox_xyxy = tracked_dets[:,:4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)
           
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    random_color_list()
        
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()



import hydra
import torch
import cv2
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import subprocess
import numpy as np
import json
import warnings
from functions import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Setting parameters and variables
input_dir = r"..\data\VID-20220108-WA0006.mp4"
output_dir = r'out\third_eye_tracker.mp4'

with open("obs_state.json", "w") as outfile:
    outfile.write(json.dumps({"state": [0, 0, 0]}))

roi = 0.35
ext_roi = 0.1
conf_threshold = 0.3
size_threshold = {'car': 60, 'person': 40, 'motorcycle': 40, 'truck': 80, 'bicycle': 40, 'parking meter': 0, 'cow': 0, 'dog': 0}
size_threshold_outside_roi = {'car': 220, 'person': 80, 'motorcycle': 150, 'truck': 250, 'bicycle': 150, 'parking meter': 0, 'cow': 0, 'dog': 0}

warn_avg_size = 30
del_angle_threshold = 0.2
del_area_threshold = 0.35
crowd_threshold = 10

# Initialize DeepSORT
cfg = get_config()
cfg.merge_from_file(config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

def alert_func():
    subprocess.Popen("python alert_3dsound.py", shell=True)

def new_obs(label, warn_avg_size):
    return {'label': label, 'warn_avg_size': warn_avg_size, 'area_hist': [], 'del_area': 0, 'del_angle': 0, 'warning': [False]*warn_avg_size}

def get_del(hist):
    if len(hist) < 2:
        return 0
    return hist[-1] - hist[-2]

def draw_ROI(frame, roi, ext_roi):
    height, width = frame.shape[:2]
    left = int(width * roi)
    right = int(width * (1 - roi))
    left_ext = int(width * ext_roi)
    right_ext = int(width * (1 - ext_roi))
    cv2.line(frame, (left, 0), (left, height), (255, 0, 0), 2)
    cv2.line(frame, (right, 0), (right, height), (255, 0, 0), 2)
    return frame, left, right

def detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db):
    count_of_obstacles = 0
    mid = int((left + right) / 2)
    height, width = frame.shape[:2]
    if len(outputs) > 0:
        for j, (output, conf) in enumerate(zip(outputs, confs)):
            label = names[int(output[5])]
            if conf > conf_threshold and (label in obstacles):
                bboxes = output[0:4]
                x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                id = output[4]
                xc = int(x1 + (x2 - x1) / 2)
                yc = int(y1 + (y2 - y1) / 2)
                area = int(((x2 - x1) * (y2 - y1)) / 100)
                color = (0, 255, 0)
                left_ext, right_ext = int(ext_roi * width), int((1 - ext_roi) * width)
                if area > size_threshold[label] and (xc >= left_ext and xc <= right_ext):
                    if id not in database.keys():
                        database[id] = new_obs(label, warn_avg_size)
                    database[id]['area_hist'].append(area)
                    database[id]['del_area'] = get_del(database[id]['area_hist'])
                    database, frame = update_angle(database, id, xc, yc, frame)
                    if x2 >= left and x1 <= right:
                        count_of_obstacles += 1
                        color = (0, 255, 255)
                        if database[id]["del_angle"] < del_angle_threshold and database[id]['del_area'] > del_area_threshold:
                            database[id]["warning"].append(True)
                        else:
                            database[id]["warning"].append(False)
                    elif (database[id]['del_angle'] < 0 and database[id]['del_area'] > 0) and area >= size_threshold_outside_roi[label]:
                        database[id]["warning"].append(False)
                    else:
                        database[id]["warning"].append(False)
                    if np.sum(database[id]["warning"]) >= (warn_avg_size / 2):
                        color = (0, 0, 255)
                        warn_db[id] = [xc, yc]
                        if x2 <= mid and x2 >= left:
                            obs[0] += 1
                        elif x1 >= mid and x1 <= right:
                            obs[1] += 1
                        else:
                            obs[2] += 1
                    temp = np.round(database[id]['del_area'], 2)
                    disp = f'{id} {temp} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, disp, (x1 - 1, y1 - 1), font, 0.5, (255, 0, 255), 1)
                    refpt = (int(width / 2), int(height))
                    cv2.line(frame, refpt, (int(width / 2), 0), (0, 0, 234), 2)
                    cv2.arrowedLine(frame, refpt, (xc, yc), (123, 232, 324), 1)
    obs_current = [0, 0, 0]
    for i in range(3):
        if obs[i] >= 1:
            warn[i] += "WARNING"
            obs_current[i] = 1
    cv2.putText(frame, str(obs[0]) + warn[0], (left, frame.shape[:2][0]), font, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, str(obs[2]) + warn[2], (mid, frame.shape[:2][0]), font, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, str(obs[1]) + warn[1], (right, frame.shape[:2][0]), font, 0.5, (0, 0, 255), 2)
    warn_color = (0, 255, 0)
    global obs_hist, warn_count
    obs_current.append(count_of_obstacles >= crowd_threshold)
    if obs_hist != obs_current:
        send_state(obs_current)
        print(obs_current)
        warn_count += 1
    if obs_current[2] > 0:
        warn_color = (0, 0, 255)
        if obs_hist[2] != obs_current[2]:
            pass
    elif obs_current[0] > 0 or obs_current[1] > 0:
        warn_color = (0, 255, 255)
    else:
        warn_color = (0, 255, 0)
    obs_hist = obs_current
    cv2.circle(frame, (int(0.95 * width), int(0.90 * height)), int(0.01 * (height + width)), warn_color, -1)
    return database, frame

def init_tracker():
    global tracker
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

rand_color_list = []

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (x1, y1, x2, y2, id)
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = (x1 + t_size[0], y1 - t_size[1] - 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), c2, color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        label = names[cat]
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = (x1 + t_size[0], y1 + t_size[1] + 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), c2, color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]
        return img

    @torch.no_grad()
    def predict(self, source=None, stream=False):
        model = self.model
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt, vid_stride=self.args.vid_stride)
        self.model.eval()
        seen, dt = 0, [Profile(), Profile(), Profile()]
        curr_frames, prev_frames = [], []
        init_tracker()
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            with dt[0]:
                im = self.preprocess(im)
            with dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=False)
            with dt[2]:
                preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, classes=self.args.classes, agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
            for i, det in enumerate(preds):
                seen += 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                curr_frames.append(im0)
                s += '%gx%g ' % im.shape[2:]
                annotator = self.get_annotator(im0)
                if det is not None and len(det):
                    det[:, :4] = ops.scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    annotator.boxes(det[:, :4], det[:, 5], det[:, 4], names=self.model.names, line_thickness=self.args.line_thickness)
                    if stream:
                        for j, (output, conf) in enumerate(zip(det[:, :4], det[:, 4])):
                            bboxes = det[:, :4]
                            confs = det[:, 4]
                            cls = det[:, 5]
                            names = model.names
                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            if len(outputs) > 0:
                                bbox_xyxy = outputs[:, :4]
                                identities = outputs[:, -1]
                                categories = outputs[:, -2]
                                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, names)
                cv2.imshow(str(p), im0)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_dir / f'{p.stem}.jpg', im0)
                    else:
                        vid_writer.write(im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
            prev_frames = curr_frames
            curr_frames = []
        if save_img:
            print(f' {s}Done. ({t3 - t2:.3f}s)')
            print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    with torch.no_grad():
        detect('yolov8.pt', input_dir)
