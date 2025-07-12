import os
import logging
import numpy as np
import cv2
import torch
from torchvision.ops import masks_to_boxes
from typing import Generator
from dataclasses import dataclass

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor


@dataclass
class MobileSAMV2Config:
    image_size: int = 320                 # Size of the input image for segmentation
    confidence_threshold: float = 0.5     # Confidence threshold for object detection
    iou: float = 0.3                      # Intersection over Union threshold for filtering detections
    remove_fully_contained: bool = False  # Whether to remove fully contained masks from segmentation
    batch_size: int = 256                 # Batch size for processing images


class MobileSAMV2:
    """
    This class provides functionality to segment all objects in an input image using MobileSAMv2.
    """

    MOBILE_SAM_V2_PATH = '/MobileSAM/MobileSAMv2'
    PROMPT_GUIDED_PATH = os.path.join(MOBILE_SAM_V2_PATH, 'PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt')
    OBJ_MODEL_PATH = os.path.join(MOBILE_SAM_V2_PATH, 'weight/ObjectAwareModel.pt')
    IMAGE_ENCODER_CHECKPOINT_PATH = os.path.join(MOBILE_SAM_V2_PATH, 'weight/l2.pt')

    COLORS = [
        [0.216, 0.494, 0.722],  # blue
        [0.894, 0.102, 0.110],  # red
        [0.302, 0.686, 0.290],  # green
        [0.596, 0.306, 0.639],  # purple
        [1.000, 0.498, 0.000],  # orange
        [1.000, 1.000, 0.200],  # yellow
        [0.651, 0.337, 0.157],  # brown
        [0.969, 0.506, 0.749],  # pink
        [0.600, 0.600, 0.600],  # gray
        [0.121, 0.470, 0.705],  # another blue
    ]

    def __init__(self, config: MobileSAMV2Config = MobileSAMV2Config()):
        """
        Initializes the MobileSAMV2 model.
        """
        # Set parameters
        self.image_size = config.image_size
        self.confidence_threshold = config.confidence_threshold
        self.iou = config.iou
        self.remove_fully_contained = config.remove_fully_contained
        self.batch_size = config.batch_size

        # Create model and predictor
        self.mobilesamv2, self.obj_aware_model = self.create_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mobilesamv2.to(device=self.device)
        self.mobilesamv2.eval()
        self.predictor = SamPredictor(self.mobilesamv2)

    def enable_logging(self, enable: bool):
        """
        Enable or disable logging for the model.
        """
        logging.getLogger("ultralytics").setLevel(
            logging.INFO if enable else logging.ERROR)

    def assert_image_format(self, image: np.ndarray):
        assert isinstance(image, np.ndarray), "Input image must be a NumPy array."
        assert image.ndim == 3, "Input image must be a 3D array (H, W, C)."
        assert image.shape[2] == 3, "Input image must have 3 channels (RGB)."

    def create_model(self):
        obj_aware_model = ObjectAwareModel(self.OBJ_MODEL_PATH)
        mobilesamv2 = sam_model_registry['vit_h']()
        prompt_guided_decoder = sam_model_registry['PromptGuidedDecoder'](self.PROMPT_GUIDED_PATH)
        mobilesamv2.prompt_encoder = prompt_guided_decoder['PromtEncoder']
        mobilesamv2.mask_decoder = prompt_guided_decoder['MaskDecoder']
        mobilesamv2.image_encoder = sam_model_registry['efficientvit_l2'](self.IMAGE_ENCODER_CHECKPOINT_PATH)
        return mobilesamv2, obj_aware_model

    @staticmethod
    def batch_iterator(batch_size: int, *args) -> Generator[list[any], None, None]:
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "Batched iteration must have inputs of all the same size."

        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]

    @staticmethod
    def remove_fully_contained_masks(masks, threshold=0.95):
        """
        Removes masks that are fully covered by another mask.

        Args:
            masks (Tensor): [N, H, W] boolean tensor.
            threshold (float): ratio of overlap to mask area for removal.

        Returns:
            Tensor: keep (bool): [N] boolean tensor indicating which masks to keep.
        """
        N = masks.shape[0]
        keep = torch.ones(N, dtype=torch.bool, device=masks.device)

        mask_areas = masks.flatten(1).sum(dim=1)

        for i in range(N):
            if not keep[i]:
                continue
            for j in range(N):
                if i == j or not keep[j]:
                    continue
                intersection = (masks[i] & masks[j]).sum().item()
                if intersection / mask_areas[i].item() >= threshold:
                    # If i is fully covered by j, remove i
                    keep[i] = False
                    break

        return keep

    def get_prediction(self, image: np.ndarray) -> list:
        """
        Segments all detectable regions in the input image and returns predictions.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            list: A list with one dictionary containing keys: 'boxes', 'scores', 'labels', 'masks'
        """
        # Assert shape and channels
        self.assert_image_format(image)

        obj_results = self.obj_aware_model(
            image, device=self.device, retina_masks=True,
            imgsz=self.image_size, conf=self.confidence_threshold, iou=self.iou
        )

        self.predictor.set_image(image)

        # Get bounding boxes and associated scores and labels
        input_boxes = obj_results[0].boxes.xyxy  # (N, 4)
        scores = obj_results[0].boxes.conf        # (N,)
        labels = obj_results[0].boxes.cls.int()   # (N,)

        # Transform boxes for SAM
        input_boxes = input_boxes.cpu().numpy()
        input_boxes = self.predictor.transform.apply_boxes(input_boxes, self.predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()

        # Prepare embeddings
        sam_mask = []
        image_embedding = self.predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, self.batch_size, dim=0)
        prompt_embedding = self.mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, self.batch_size, dim=0)

        # Run segmentation in batches
        for (boxes,) in self.batch_iterator(self.batch_size, input_boxes):
            with torch.no_grad():
                image_embedding_batch = image_embedding[0:boxes.shape[0], :, :, :]
                prompt_embedding_batch = prompt_embedding[0:boxes.shape[0], :, :, :]

                sparse_embeddings, dense_embeddings = self.mobilesamv2.prompt_encoder(
                    points=None, boxes=boxes, masks=None
                )

                low_res_masks, _ = self.mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding_batch,
                    image_pe=prompt_embedding_batch,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )

                low_res_masks = self.predictor.model.postprocess_masks(
                    low_res_masks, self.predictor.input_size, self.predictor.original_size
                )

                binary_masks = (low_res_masks > self.mobilesamv2.mask_threshold).float()
                sam_mask.append(binary_masks.squeeze(1))

        # Concatenate all predicted masks
        pred_masks = torch.cat(sam_mask, dim=0).bool()

        if self.remove_fully_contained:
            keep = self.remove_fully_contained_masks(pred_masks)
            pred_masks = pred_masks[keep]
            scores = scores[keep]
            labels = labels[keep]

        # Get boxes from masks
        pred_boxes = masks_to_boxes(pred_masks)  # (N, 4)

        return {
            "boxes": pred_boxes,
            "scores": scores,
            "labels": labels,
            "masks": pred_masks
        }

    @classmethod
    def get_segmented_image_from_masks(cls, image, masks):
        """
        Returns a segmented image from the given masks.
        """
        if masks.shape[0] == 0:
            # Return blank transparent image if no masks
            h, w = image.shape[:2]
            return np.zeros((h, w, 4), dtype=np.float32)

        h, w = masks.shape[1], masks.shape[2]
        img = np.ones((h, w, 3), dtype=np.float32)  # RGB white background

        # Sort masks by area (largest first)
        areas = torch.sum(masks, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        masks_sorted = masks[sorted_indices]

        for i in range(masks_sorted.shape[0]):
            m = masks_sorted[i].bool().cpu().numpy()
            color_mask = cls.COLORS[i % len(cls.COLORS)]
            img[m] = color_mask

        return img  # shape (H, W, 3)

    def test(self, image_path: str, output_path: str = './output.jpg') -> None:
        """
        Test the segmentation on a saved image.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the output segmented image.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_masks = self.get_prediction(image)['masks']
        segmented_image = self.get_segmented_image_from_masks(image, all_masks)
        
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    # Don't put these imports outside of main.
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for saving figures
    from matplotlib import pyplot as plt

    # Initialize the MobileSAMV2 model and test on a single image
    mobile_sam_v2 = MobileSAMV2()
    test_image_path = '/MobileSAM/MobileSAMv2/test_images/1.jpg'
    output_image_path = './output.jpg'
    mobile_sam_v2.test(test_image_path, output_image_path)
