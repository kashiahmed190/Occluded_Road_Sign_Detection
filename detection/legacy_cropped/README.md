# Detection — legacy cropped-patch pipeline (superseded)

Used synthesized full-frame bounding boxes (since the cropped patches had
no finer box to recover) and a simplified "does the correct class appear
anywhere in the image" detection metric rather than real IoU-based
matching. Kept for reference. `PIPLINEdetect.py`/`Evalfinal.py` are renamed
copies of `run_full_inpainting_detection_pipeline.py`/`evaluate_detection.py`
from partway through development.
