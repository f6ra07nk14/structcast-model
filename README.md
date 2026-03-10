# StructCast-Model

Construct neural network models and training workflows by structcast package.

uv sync --extra torch-cu130 --extra mlflow --extra flops 
scm torch create model cfg/models/ConvNeXtV2.yaml
scm torch create model cfg/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' -s 'image: [3, 224, 224]' --backend pytorch
scm torch calflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' -s 'image: [3, 224, 224]'
scm torch create model cfg/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' 
scm format cfg/datasets/default_timm.yaml -o dataset_train.yaml -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100,  label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
scm format cfg/datasets/default_timm.yaml -o dataset_valid.yaml -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
scm torch train 'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' -s 'image: [3, 224, 224]' -d cuda --ema cfg/others/ema.yaml -l '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' -m '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' -b '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' -c cfg/others/compile_default.yaml -e 5 -t dataset_train.yaml -v dataset_valid.yaml -f 1 -LC ce_loss -LC val_ce_loss -HC acc1 -HC val_acc1 -HC acc5 -HC val_acc5 -SC val_acc1 --matmul-precision high -E Test -A model.py -A cfg/others/ema.yaml -A loss.py -A metric.py -A backward.py -A cfg/others/compile_default.yaml -A dataset_train.yaml -A dataset_valid.yaml


## Roadmap

- [x] PyTorch model construction from YAML configuration file.
- [x] Create PyTorch training workflow from YAML configuration file.
- [ ] Jax model construction from YAML configuration file.  
- [ ] Create Jax training workflow from YAML configuration file.
- [ ] TensorFlow model construction from YAML configuration file.
- [ ] Create TensorFlow training workflow from YAML configuration file.
