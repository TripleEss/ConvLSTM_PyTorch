# ConvLSTM_PyTorch

## Prepare data

```bash
curl -o mnist_test_seq.npy http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
```

## Train

```bash
python train.py --epochs 20--save-model-path './weight' --save-freq 1 --log-dir './log'
```
