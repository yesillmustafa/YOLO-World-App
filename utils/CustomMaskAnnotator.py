import cv2
import numpy as np

class CustomMaskAnnotator:
    def __init__(self):
        pass

    def annotate(self, image, detections):
        if detections.mask is None:
            return image

        mask_shape = detections.mask.shape[1:]  # İlk boyutu (kanal sayısı) almıyoruz
        height, width = mask_shape

        # Dışındaki alanı beyaz yapacak bir maske oluştur
        white_mask = np.ones((height, width), dtype=np.uint8) * 255

        for i in range(height):
            for j in range(width):
                if detections.mask[:, i, j].any():  # Herhangi bir kanalda pikselin True olduğunu kontrol et
                    white_mask[i, j] = 0

        # Maskeyi tersine çevir
        black_mask = cv2.bitwise_not(white_mask)

        # Siyah zemin üzerinde sadece tespit edilen nesneyi gösterecek şekilde bir görüntü oluştur
        result = cv2.bitwise_and(image, image, mask=black_mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result
