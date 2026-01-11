
import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_thermal_cheetah_poly"

        # ---------------- model config ---------------- #
        self.num_classes = 3
        
        # ---------------- polygon toggle ---------------- #
        self.use_polygon = True     # <--- ENABLE POLYGON SUPPORT
        
        # ---------------- dataset config ---------------- #
        # Use absolute path or relative to d:\YOLOX
        self.data_dir = "datasets/Thermal Cheetah.v1-square.coco"
        self.train_ann = "instances_train2017_poly.json"
        self.val_ann = "instances_val2017_poly.json"
        self.test_ann = "instances_test2017_poly.json"
        
        # Helper to set correct folder names if Roboflow uses different names
        self.train_name = "train"
        self.val_name = "valid"
        self.test_name = "test"

        # ---------------- training settings ---------------- #
        self.max_epoch = 10         # Small test run
        self.print_interval = 1
        self.eval_interval = 5
        self.input_size = (640, 640)
        self.test_size = (640, 640)
