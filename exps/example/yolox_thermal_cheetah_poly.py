
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
        self.train_name = "train2017"
        self.val_name = "val2017"
        self.test_name = "test2017"

        # ---------------- training settings ---------------- #
        self.max_epoch = 10         # Small test run
        self.print_interval = 1
        self.eval_interval = 5
        self.input_size = (640, 640)
        self.test_size = (640, 640)

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                use_polygon=self.use_polygon,
            ),
            cache=cache,
            cache_type=cache_type,
            use_polygon=self.use_polygon,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_name if not testdev else self.test_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            use_polygon=self.use_polygon,
        )
