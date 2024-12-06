import numpy as np

loaded_data = np.load("target_gesture_embed_log.npy", allow_pickle=True)
tmp = np.mean(loaded_data , axis=0)
print(tmp)
print(len(loaded_data))  # Đây là một danh sách các np.ndarray