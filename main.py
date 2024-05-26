import cv2
import numpy as np
import torch
import supervision as sv
from tqdm import tqdm
from inference.models import YOLOWorld
from utils.EfficientSAM import load, inference_with_boxes
from utils.my_utils import draw_text_on_image, calculate_end_frame_index
from utils.CustomMaskAnnotator import CustomMaskAnnotator
from typing import List

############ PREFERENCES ############
CONFIDENCE = 0.02
IOU_THRESHOLD = 0.5

CATEGORIES = ["aircraft"]

FILE_NAME = "fighter.mp4"

CROPPING_MODE = True
#####################################

############## SETTINGS #############
if(FILE_NAME.endswith(".mp4")):
    SOURCE_FILE_PATH = f"assets/videos/{FILE_NAME}"
    TARGET_FILE_PATH = f"results/videos/r_{FILE_NAME}"
else:
    SOURCE_FILE_PATH = f"assets/images/{FILE_NAME}"
    TARGET_FILE_PATH = f"results/images/r_{FILE_NAME}"
    IMG_ARRAY = cv2.imread(SOURCE_FILE_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(device=DEVICE)
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
if(not FILE_NAME.endswith(".mp4") and CROPPING_MODE == True):
    MASK_ANNOTATOR = CustomMaskAnnotator()
else:
    MASK_ANNOTATOR = sv.MaskAnnotator()
#####################################

def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}" 
            if with_confidence 
            else f"{categories[class_id]}"
        )
        for class_id, confidence 
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = MASK_ANNOTATOR.annotate(input_image,detections)
    if(not CROPPING_MODE):
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image,detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image,detections, labels=labels)
    return annotated_image

def process_image(
        input_image: np.ndarray,
        categories: np.ndarray,
        confidence_value: float = 0.3,
        iou_threshold_value: float = 0.5,
        with_segmentation: bool = True,
        with_confidence: bool = True,
        with_class_agnostic_nms: bool = False,
) -> np.ndarray:
    YOLO_WORLD_MODEL.set_classes(CATEGORIES)
    results = YOLO_WORLD_MODEL.infer(input_image,confidence=confidence_value)
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(
        threshold=iou_threshold_value,
        class_agnostic=with_class_agnostic_nms)

    if with_segmentation:
        detections.mask = inference_with_boxes(
            image=input_image,
            xyxy=detections.xyxy,
            model=EFFICIENT_SAM_MODEL,
            device=DEVICE
        )
    
    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence
    )
    final_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    text = f"Confidence: {CONFIDENCE}\nIOU Threshold: {IOU_THRESHOLD}\nCropping Mode: {CROPPING_MODE}"
    image = draw_text_on_image(frame=final_image,text=text)
    image.save(TARGET_FILE_PATH)
    image.show()

def process_video(
        input_video: str,
        categories: np.ndarray,
        confidence: float = 0.3,
        iou_threshold: float = 0.5,
        with_segmentation: bool = True,
        with_confidence: bool = True,
        with_class_agnostic_nms: bool = False
) -> str:
    YOLO_WORLD_MODEL.set_classes(CATEGORIES)
    video_info = sv.VideoInfo.from_video_path(input_video)
    end_frame_index = calculate_end_frame_index(input_video)
    frame_generator  = sv.get_video_frames_generator(
        source_path=input_video,
        end=end_frame_index
    )
    with sv.VideoSink(TARGET_FILE_PATH,video_info=video_info) as sink:
        for _ in tqdm(range(end_frame_index), desc="Processing video.."):
            frame = next(frame_generator)
            results = YOLO_WORLD_MODEL.infer(frame, confidence=confidence)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=frame,
                    xyxy=detections.xyxy,
                    model=EFFICIENT_SAM_MODEL,
                    device=DEVICE
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = annotate_image(
                input_image=frame,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence
            )
            text = f"Confidence: {CONFIDENCE}\nIOU Threshold: {IOU_THRESHOLD}\nCropping Mode: {CROPPING_MODE}"
            scene = draw_text_on_image(frame,text)
            frame = np.array(scene)
            sink.write_frame(frame)
        
# Main
if(FILE_NAME.endswith(".mp4")):
    process_video(SOURCE_FILE_PATH,CATEGORIES,CONFIDENCE,IOU_THRESHOLD)
else:
    process_image(IMG_ARRAY,CATEGORIES,CONFIDENCE,IOU_THRESHOLD)
