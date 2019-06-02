import argparse
import os
import pandas as pd
import mxnet as mx
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

BATCH_SIZE = 256
real_label = 2

models = {
    # 'se_resnext101_32x4d': 'se_resnext_finetune-0012.params',
    'se_resnext101_32x4d': 'se_resnext101_32x4d_ft_res-0030.params',
    'ResNet50_v2': 'renetv2_finetune-0003.params',
    'resnet50_v1d_0.86': 'resnet50_v1prunned_finetune-0010.params'
}


class TestAntispoofDataset(mx.gluon.data.Dataset):
    def __init__(
            self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        image_info = self.paths[index]

        img = image.imread(image_info['path'])
        if self.transform is not None:
            img = self.transform(img)

        return img, index  # image_info['id'], image_info['frame']

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    parser.add_argument('--path-submission-csv', type=str, required=True)
    args = parser.parse_args()

    # prepare image paths
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [
        {
            'id': row.id,
            'frame': row.frame,
            'path': os.path.join(path_test_dir, row.path)
        } for _, row in test_dataset_paths.iterrows()]

    # prepare dataset and loader
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = TestAntispoofDataset(
        paths=paths, transform=data_transforms)
    dataloader = gluon.data.DataLoader(
        image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_gpus = 1
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    classes = 4

    model_name = 'se_resnext101_32x4d'
    model_weights = models[model_name]
    # load model
    finetune_net = get_model(model_name, pretrained=False)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(classes)
    finetune_net.load_parameters(model_weights, ctx=ctx)
    finetune_net.hybridize()

    # predict
    samples, frames, probabilities = [], [], []

    for batch, index in dataloader:
        data = gluon.utils.split_and_load(batch, ctx_list=ctx, batch_axis=0, even_split=False)
        out = finetune_net(data[0])
        prob = nd.softmax(out)
        real_probability = prob[:, real_label]
        
        samples.extend([paths[i]['id'] for i in index.asnumpy()])
        frames.extend([paths[i]['frame'] for i in index.asnumpy()])
        probabilities.extend(1 - real_probability.asnumpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})

    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
    predictions[['id', 'prediction']].to_csv(
        args.path_submission_csv, index=False)
