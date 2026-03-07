import torch
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.utils import register_all_modules
from mmengine.dataset import Compose


class ActionDetector:
    def __init__(self):
        register_all_modules(init_default_scope=False)

        self.config = "mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
        self.checkpoint = "weights/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
        self.label_file = "mmaction2/tools/data/kinetics/label_map_k400.txt"

        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = init_recognizer(self.config, self.checkpoint, device=self.device)

        with open(self.label_file, "r") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        self.test_pipeline = Compose([
            dict(type="OpenCVInit", io_backend="disk"),
            dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
            dict(type="OpenCVDecode"),
            dict(type="Resize", scale=(-1, 256)),
            dict(type="CenterCrop", crop_size=224),
            dict(type="FormatShape", input_format="NCHW"),
            dict(type="PackActionInputs"),
        ])

    def predict(self, video_path: str) -> str:
        result = inference_recognizer(
            self.model,
            video_path,
            test_pipeline=self.test_pipeline,
        )

        scores = result.pred_score.detach().cpu()
        top_score, top_idx = torch.max(scores, dim=0)
        top_label = self.labels[top_idx].lower()

        fight_keywords = [
            "kick", "punch", "boxing", "wrestling",
            "slap", "hit", "karate", "taekwondo"
        ]
        fall_keywords = ["fall", "slip", "stumble", "collapse"]

        if any(word in top_label for word in fight_keywords):
            return "fight"
        if any(word in top_label for word in fall_keywords):
            return "fall"
        return "normal"


detector = ActionDetector()