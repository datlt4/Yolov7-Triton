# request-yolov4-tensorrtx.py
from audioop import reverse
from cProfile import label
import pdb
from turtle import color
from unittest import result
import numpy as np
import sys
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.utils.shared_memory as shm
from abc import ABC, abstractmethod
import requests


class bbox_t:
    def __init__(self, **kwargs):
        self.prob = kwargs["prob"]
        self.obj_id = int(kwargs["obj_id"])
        self.x = kwargs["x"]
        self.y = kwargs["y"]
        self.w = kwargs["w"]
        self.h = kwargs["h"]

    def __str__(self):
        return f"x={self.x}, y={self.y}, w={self.w}, h={self.h}, prob={self.prob}, obj_id={self.obj_id}"
    
    def to_json(self, class_id):
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h), "prob": round(float(self.prob), 2), "obj_id": class_id[self.obj_id]}


class TritonClient(ABC):
    def __init__(self, url="127.0.0.1:8001", verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None):
        super().__init__()
        self.define_triton_client(
            url, verbose, ssl, root_certificates, private_key, certificate_chain)

    def define_triton_client(self, url="127.0.0.1:8001", verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None):
        self.triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose, ssl=ssl, root_certificates=root_certificates, private_key=private_key, certificate_chain=certificate_chain)
        self.inputs = []
        self.outputs = []

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def triton_infer(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass


class TritonClientYolov7(TritonClient):
    def __init__(self, grpc_url="127.0.0.1:8001", http_url="127.0.0.1:8000", verbose=False, ssl=False, root_certificates=None, private_key=None, certificate_chain=None):
        super().__init__(grpc_url, verbose, ssl,
                         root_certificates, private_key, certificate_chain)
        self.obj_threshold = 0.35
        self.nms_threshold = 0.65
        self.IMAGE_WIDTH = 608
        self.IMAGE_HEIGHT = 608

        self.inputs.append(grpcclient.InferInput(
            "images", [1, 3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput("output"))
        # r = requests.get(f"http://{http_url}/v2/models/yolov7-trt")
        # assert r.status_code == 200, "Request for unknown model: `yolov7-tr` is not found"

        self.class_id = {0: "Person", 1: "Face", 2: "Vehicle",
                         3: "Weapon", 4: "NumberPlate", 5: "LargeVehicle", 6: "Bike"}
        self.class_color = {0: (235, 64, 52), 1: (161, 123, 35), 2: (52, 105, 17), 3: (
            53, 122, 84), 4: (37, 87, 115), 5: (60, 40, 128), 6: (140, 35, 98)}

    def preprocess(self, **kwargs):
        """Preprocess an image before TRT YOLO inferencing.
        # Args
            img: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
        # Returns
            preprocessed img: float32 numpy array of shape (3, H, W)
        """
        if "img" in kwargs.keys():
            img = kwargs["img"]
        else:
            return None

        if "input_shape" in kwargs.keys():
            input_shape = kwargs["input_shape"]
        else:
            return img

        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
        else:
            new_w = int(img_w * new_h / img_h)

        resized = cv2.resize(img, (new_w, new_h))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        resized /= 255.0
        resized = resized.transpose((2, 0, 1))

        img = np.full(
            (3, input_shape[0], input_shape[1]), 0.5, dtype=np.float32)
        img[:, offset_h:(offset_h + new_h),
            offset_w:(offset_w + new_w)] = resized

        return img

    def inference(self, **kwargs):
        if "img" in kwargs.keys():
            input_image_buffer = kwargs["img"]
        else:
            return None
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        self.inputs[0].set_data_from_numpy(input_image_buffer)
        results = self.triton_client.infer(
            model_name="yolov7-trt", inputs=self.inputs, outputs=self.outputs, client_timeout=None)
        return results.as_numpy("output")

    def postprocess(self, **kwargs):
        if "raw_shape" in kwargs.keys():
            raw_shape = kwargs["raw_shape"]
        else:
            return []
        ratio = (float(raw_shape[1]) / float(self.IMAGE_WIDTH)) if (float(raw_shape[1]) / float(self.IMAGE_WIDTH)
                                                                    > float(raw_shape[0]) / float(self.IMAGE_HEIGHT)) else (float(raw_shape[0]) / float(self.IMAGE_HEIGHT))
        if "outputs" in kwargs.keys():
            detections = kwargs["outputs"]
        else:
            return []
        result = []
        detection = detections[0]
        detection = detection[detection[:, 4] > self.obj_threshold]

        for r in detection:
            obj_id = np.argmax(r[5:])
            prob = r[4] * r[5 + obj_id]
            x = (r[0] - r[2] / 2.0) * ratio
            y = (r[1] - r[3] / 2.0) * ratio
            w = r[2] * ratio
            h = r[3] * ratio
            result.append(bbox_t(x=x, y=y, w=w, h=h, obj_id=obj_id, prob=prob))

        result.sort(reverse=True, key=lambda x: x.prob)
        result = self.NMS(result)
        return result

    def NMS(self, detection):
        for i in range(len(detection)):
            for j in range(i+1, len(detection)):
                if detection[i].obj_id == detection[j].obj_id:
                    iou = self.IOU(detection[i], detection[j])
                    if iou > self.nms_threshold:
                        detection[j].prob = 0

        for i in range(len(detection)-1, -1, -1):
            if detection[i].prob == 0:
                detection.pop(i)
        return detection

    def IOU(self, det_a, det_b):
        center_a = (det_a.x + det_a.w / 2, det_a.y + det_a.h / 2)
        center_b = (det_b.x + det_b.w / 2, det_b.y + det_b.h / 2)
        left_up = (min(det_a.x, det_b.x), min(det_a.y, det_b.y))
        right_down = (max(det_a.x + det_a.w, det_b.x + det_b.w),
                      max(det_a.y + det_a.h, det_b.y + det_b.h))
        distance_d = (center_a[0] - center_b[0]) ** 2 + \
            (center_a[1] - center_b[1]) ** 2
        distance_c = (left_up[0] - right_down[0]) ** 2 + \
            (left_up[1] - right_down[1]) ** 2
        inter_l = det_a.x if (det_a.x > det_b.x) else det_b.x
        inter_t = det_a.y if (det_a.y > det_b.y) else det_b.y
        inter_r = (det_a.x + det_a.w) if (det_a.x + det_a.w <
                                          det_b.x + det_b.w) else (det_b.x + det_b.w)
        inter_b = (det_a.y + det_a.h) if (det_a.y + det_a.h <
                                          det_b.y + det_b.h) else (det_b.y + det_b.h)
        if ((inter_b < inter_t) or (inter_r < inter_l)):
            return 0
        inter_area = (inter_b - inter_t) * (inter_r - inter_l)
        union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area
        if (union_area == 0):
            return 0
        elif True:
            return inter_area / union_area - distance_d / distance_c
        else:
            return inter_area / union_area - distance_d / distance_c

    def plot_one_box(self, r, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.0005 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def triton_infer(self, image):
        kwargs = {"img": image, "input_shape": [
            self.IMAGE_HEIGHT, self.IMAGE_WIDTH], "raw_shape": [image.shape[0], image.shape[1]]}
        processed_data = self.preprocess(**kwargs)
        kwargs.update({"img": processed_data})
        outputs = self.inference(**kwargs)
        kwargs.update({"outputs": outputs})
        result = self.postprocess(**kwargs)
        # for r in result:
        #     self.plot_one_box(r=r, img=image, color=self.class_color[r.obj_id], label=str(
        #         self.class_id[r.obj_id]) + " " + str(int(r.prob * 100)) + "%")
        # return image, result
        return result


if __name__ == "__main__":
    try:
        image = cv2.imread(
            "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/pose-test/person_car.png")
        client_yolov7 = TritonClientYolov7()
        image, result = client_yolov7.triton_infer(image)

        cv2.imwrite("result.png", image)
    except requests.exceptions.ConnectionError as e:
        print(e)
    except AssertionError as e:
        print(e)
