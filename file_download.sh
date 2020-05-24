mkdir data
cd data

# trainデータは容量が大きいので今回は使用しない
# wget -q http://images.cocodataset.org/zips/train2017.zip
# unzip train2017.zip

wget -q http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip