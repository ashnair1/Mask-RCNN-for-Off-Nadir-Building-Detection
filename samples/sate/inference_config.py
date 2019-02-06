from sate import CocoConfig as Config


class InferenceConfig(Config):
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    

    # # Number of classes (including background)
    # NUM_CLASSES = 1 + 1  # building detection

    # # Use smaller anchors because our images and objects are small.
    # #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)



