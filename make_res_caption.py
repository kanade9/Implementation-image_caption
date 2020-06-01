import os
import json
import hydra

base_dir = str(os.getcwd())
json_dir = base_dir+'/data/annotations/captions_val2014.json'

output_test_dir = base_dir+'/data/annotations/another/captions_test2014.json'

if __name__ == "__main__":

    with open(json_dir, encoding='utf-8') as f:
        data=json.loads(f.read())

        for key in data['images']:
            img_file_name=key['file_name']
            img_file_path=base_dir+'/data/val2014/'+img_file_name




@hydra.main(config_path="conf/config.yaml")
def make_caption(img_path):
    