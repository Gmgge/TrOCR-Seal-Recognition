import os
import codecs
import argparse
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr vocab生成')
    parser.add_argument('--cust_vocab', default="./cust-data/vocab.txt", type=str, help="自定义vocab文件生成")
    parser.add_argument('--dataset_path', default="./dataset/train/", type=str, help="自定义训练数字符集根路径")
    args = parser.parse_args()
    vocab = set()
    dateset_files = os.listdir(args.dataset_path)
    for one_file in tqdm(dateset_files):
        if one_file.endswith(".txt"):
            label_path = os.path.join(args.dataset_path, one_file)
            with codecs.open(label_path, encoding='utf-8') as f:
                txt = f.read().strip()
            vocab.update(txt)
    root_path = os.path.split(args.cust_vocab)[0]
    os.makedirs(root_path, exist_ok=True)
    with open(args.cust_vocab, 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(vocab)))
