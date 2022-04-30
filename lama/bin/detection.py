#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback
import numpy as np

from saicinpainting.evaluation.utils import move_to_device

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)




@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        # 输出mask
        cfg = get_cfg()   # get a fresh new config

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = predict_config.detect_threthold # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        predictor = DefaultPredictor(cfg)

        iterations = 10
        expand_ratio = 0.2
        # predict_config.indir
        for filename in os.listdir(predict_config.indir):
            img = cv2.imread(predict_config.indir+'/'+filename)
            outputs = predictor(img)
            w, h = outputs["instances"].image_size
            Fmask = np.zeros((w, h))
            if predict_config.detect_model == "segmentation":
              masks = outputs["instances"].pred_masks.cpu().numpy()
              i=0
              for data in outputs["instances"].pred_classes:  
                  num = data.item()
                  #print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
                  if str(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num]) in predict_config.category:
                    #print(predict_config.category)
                    Fmask=Fmask+masks[i]
                  i=i+1
              Fmask[Fmask>0] = 255
              #print('1', sum((map(sum, Fmask))))
              kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
              Fmask = cv2.dilate(Fmask,kernel,iterations=iterations)
              if max(map(max, Fmask)) == 0:
                print("Can not detect: ", predict_config.category)
                sys.exit()
              #print('2',sum((map(sum, Fmask))))
            else:
                # print(outputs["instances"].to("cpu"))
                i = 0
                boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
                print(boxes[0])
                #print(outputs["instances"].image_size)
                for data in outputs["instances"].pred_classes:
                  num = data.item()
                  # print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num])
                  if MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[num]==predict_config.category:
                    x1, y1, x2, y2 = boxes[i]
                    x1 = int(np.floor(x1))
                    x2 = int(np.ceil(x2))
                    y1 = int(np.floor(y1))
                    y2 = int(np.ceil(y2))
                    if expand_ratio >0:
                      x1 = max(0, int(x1-(x2-x1)*expand_ratio))
                      x2 = min(w, int(x2+(x2-x1)*expand_ratio))
                      y1 = max(0, int(y1-(y2-y1)*expand_ratio))
                      y2 = min(h, int(y2+(y2-y1)*expand_ratio))
                    #print(x1, y1, x2, y2)
                    Fmask[x1:x2, y1:y2] = 255
                    #print(Fmask[x1:x2, y1:y2])
                
                if max(map(max, Fmask)) == 0:
                  print("Can not detect: ", predict_config.category)
                  sys.exit()
            if filename[-4:] in ['.jpg', '.png']:
                cv2.imwrite(predict_config.indir+'/'+filename[:-4]+'_mask.png', Fmask)
            elif filename[-5:] in ['.jpeg']:
                cv2.imwrite(predict_config.indir+'/'+filename[:-5]+'_mask.png', Fmask)


        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
