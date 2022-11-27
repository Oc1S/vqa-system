cd apis/vqa
if [ ! -d 'data' ]; then
    mkdir data
fi
wget -c https://xaneon.oss-cn-hangzhou.aliyuncs.com/data/vocab_file2.pkl
wget -c https://xaneon.oss-cn-hangzhou.aliyuncs.com/data/vgg16.tfmodel
if [ ! -d 'models' ]; then
    mkdir models
fi
cd models
wget -c https://xaneon.oss-cn-hangzhou.aliyuncs.com/data/models/model.ckpt.data-00000-of-00001
wget -c https://xaneon.oss-cn-hangzhou.aliyuncs.com/data/models/model.ckpt.index
wget -c https://xaneon.oss-cn-hangzhou.aliyuncs.com/data/models/model.ckpt.meta
