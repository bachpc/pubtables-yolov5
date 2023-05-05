# PubTables-1M training by YOLOv5

Firstly, download PubTables-1M dataset and put it into `/PubTables-1M`.

Train and val:
```bash
python train.py --batch-size -1 --data data/custom-detection.yaml --weights yolov5m.pt --img 640 --project PubTables-1M-YOLO --name yolov5m-custom-detection --hyp data/hyps/hyp.scratch-med.yaml --epochs 15 --device 0

python val.py --batch-size 256 --data data/custom-detection.yaml --weights PubTables-1M-YOLO/yolov5m-custom-detection/weights/best.pt --project PubTables-1M-YOLO --name yolov5m-custom-detection-val --img 640 --task test --device 0

python train.py --batch-size -1 --data data/custom-structure.yaml --weights yolov5m.pt --img 640 --project PubTables-1M-YOLO --name yolov5m-custom-structure --hyp data/hyps/hyp.scratch-med.yaml --epochs 15 --device 0

python val.py --batch-size 256 --data data/custom-structure.yaml --weights PubTables-1M-YOLO/yolov5m-custom-structure/weights/best.pt --img 640 --project PubTables-1M-YOLO --name yolov5m-custom-structure-val --task test --device 0

python val_structure_grits.py --batch-size 256 --data data/custom-structure.yaml --weights PubTables-1M-YOLO/yolov5m-custom-structure/weights/best.pt --img 640 --project PubTables-1M-YOLO --name yolov5m-custom-structure-val --task test --conf-thres 0.25 --iou-thres 0.45 --device 0
```

<b>Table Detection:</b>
<table>
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Schedule</th>
      <th>AP50</th>
      <th>AP</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: right;">
      <td>YOLOv5m</td>
      <td>15 Epochs</td>
      <td>0.995</td>
      <td>0.988</td>
      <td>0.998</td>
      <td>0.999</td>
    </tr>
  </tbody>
</table>

<b>Table Structure Recognition:</b>
<table>
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Schedule</th>
      <th>AP50</th>
      <th>AP</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>GriTS<sub>Top</sub></th>
      <th>GriTS<sub>Con</sub></th>
      <th>GriTS<sub>Loc</sub></th>
      <th>Acc<sub>Con</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: right;">
      <td>YOLOv5m</td>
      <td>15 Epochs</td>
      <td>0.973</td>
      <td>0.912</td>
      <td>0.908</td>
      <td>0.950</td>
      <td>0.9826</td>
      <td>0.9828</td>
      <td>0.9751</td>
      <td>0.8128</td>
    </tr>
  </tbody>
</table>
