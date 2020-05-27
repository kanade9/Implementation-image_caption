import os
import sys
import zipfile
import urllib.request

# url = "http://images.cocodataset.org/zips/train2014.zip"
# save_path = "./data/train2014.zip"
#
# with zipfile.ZipFile(save_path) as zip_file:
#     zip_file.extractall('./data/')

import urllib.request
from colorama import init, Fore, Back, Style


# 進捗を表示させるためのコールバック関数
# イメージ：[=====>    ] 50.54% ( 1050KB )
def progress_print(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    # 100より大きいと見た目が悪いので……
    if percentage > 100:
        percentage = 100
    # バーはmax_bar個で100％とする
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '  # これで空のとこを埋める
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        Fore.LIGHTCYAN_EX,
        f'[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r',
        end=''
    )


def download(url, filepath_and_filename):
    init()
    print(Fore.LIGHTGREEN_EX, 'download from:', end="")
    print(Fore.WHITE, url)
    # コールバックでprogress_printを呼んで進捗表示
    urllib.request.urlretrieve(url, filepath_and_filename, progress_print)
    print('')  # 改行
    print(Style.RESET_ALL, end="")


url = "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
save_path = "./data/captions_train-val2014.zip"

if not os.path.exists(save_path):
    download(url, save_path)

with zipfile.ZipFile(save_path) as zip_file:
    zip_file.extractall('./data/')

url = "http://images.cocodataset.org/zips/train2014.zip"
save_path = "./data/train2014.zip"

if not os.path.exists(save_path):
    download(url, save_path)

with zipfile.ZipFile(save_path) as zip_file:
    zip_file.extractall('./data/')

url = "http://images.cocodataset.org/zips/val2014.zip"
save_path = "./data/val2014.zip"

if not os.path.exists(save_path):
    download(url, save_path)

with zipfile.ZipFile(save_path) as zip_file:
    zip_file.extractall('./data/')
