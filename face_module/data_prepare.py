import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_process import face_detecting


def data_aug(img):
    data = []
    for i in range(9):
        # Chuyển ảnh về float32
        img_aug = tf.image.convert_image_dtype(img, dtype=tf.float32)
        
        # Thay đổi độ sáng
        img_aug = tf.image.stateless_random_brightness(img_aug, max_delta=0.02, seed=(i, 1))
        
        # Thay đổi độ tương phản
        img_aug = tf.image.stateless_random_contrast(img_aug, lower=0.6, upper=1.2, seed=(i, 2))
        
        # Thay đổi độ bão hòa
        img_aug = tf.image.stateless_random_saturation(img_aug, lower=0.8, upper=1.2, seed=(i, 3))
        
        # Thay đổi độ sắc nét (JPEG quality)
        img_aug = tf.image.convert_image_dtype(img_aug, dtype=tf.uint8)
        img_aug = tf.image.stateless_random_jpeg_quality(img_aug, min_jpeg_quality=80, max_jpeg_quality=100, seed=(i, 4))
        img_aug = tf.image.convert_image_dtype(img_aug, dtype=tf.float32)
        
        # Lật ảnh
        img_aug = tf.image.stateless_random_flip_left_right(img_aug, seed=(i, 5))
        
        # Dịch chuyển (padding + cropping)
        img_aug = tf.image.stateless_random_crop(
            tf.image.resize_with_pad(img_aug, img.shape[0] + 20, img.shape[1] + 20),
            size=(img.shape[0], img.shape[1], img.shape[2]),
            seed=(i, 6)
        )
        
        # Tạo góc xoay ngẫu nhiên
        angle = tf.random.uniform([], minval=-15, maxval=15, dtype=tf.float32)
        angle_radians = angle * (np.pi / 180)

        cos_angle = tf.math.cos(angle_radians)
        sin_angle = tf.math.sin(angle_radians)
        transform_matrix = tf.stack([cos_angle, -sin_angle, 0, sin_angle, cos_angle, 0], axis=0)
        
        # Sửa transform_matrix để có kích thước [1, 8]
        transform_matrix = tf.concat([transform_matrix, tf.constant([0.0, 0.0])], axis=0)
        transform_matrix = tf.reshape(transform_matrix, [1, 8])
        
        # Xoay ảnh
        img_shape = tf.shape(img)
        img_aug = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img_aug, axis=0),
            transforms=transform_matrix,
            output_shape=[img_shape[0], img_shape[1]],
            interpolation="BILINEAR",
            fill_value=0.0  # Giá trị pixel mặc định cho vùng trống
        )
        img_aug = tf.squeeze(img_aug)
        
        # Thêm nhiễu Gaussian
        noise = tf.random.stateless_normal(shape=tf.shape(img_aug), mean=0.0, stddev=0.01, seed=(i, 7))
        img_aug = img_aug + noise
        
        # Đảm bảo giá trị ảnh nằm trong khoảng [0, 1]
        img_aug = tf.clip_by_value(img_aug, 0.0, 1.0)
        
        # Chuyển ảnh về uint8
        img_aug = tf.image.convert_image_dtype(img_aug, dtype=tf.uint8)
        
        # Thêm ảnh vào danh sách
        data.append(img_aug)
    
    return data

face = face_detecting(r'assets\\input\\Phuc.jpg')
data = data_aug(face[0])


for i, augmented_img in enumerate(data):
    plt.subplot(3, 3, i+1)
    plt.imshow(augmented_img.numpy().astype("uint8"))
    plt.axis('off')

plt.show()