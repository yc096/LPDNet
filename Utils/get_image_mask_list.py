import os

def get_image_list(image_root, mask_root=None):

    assert os.path.isdir(image_root), NotADirectoryError('image root is wrong!got image pth : {}'.format(image_root))
    save_path = os.path.abspath(os.path.join(image_root,'..','image_list.txt'))
    if mask_root != None:
        assert os.path.isdir(mask_root), NotADirectoryError('mask root is wrong!got image pth : {}'.format(mask_root))
        save_path = os.path.abspath(os.path.join(image_root,'..','image_mask_list.txt'))

    image_list = os.listdir(image_root)
    image_list.sort(key=lambda x: int(x.split('.')[0]))  # 仅对 数字.jpg 这样的命名格式排序
    if mask_root != None:
        mask_list = os.listdir(mask_root)
        mask_list.sort(key=lambda x: int(x.split('.')[0])) #仅对 数字.jpg 这样的命名格式排序

    if mask_root != None:
        f = open(save_path,mode='w+',encoding='utf-8')
        for index in range(len(image_list)):
            line = image_list[index].strip() + '\t' + mask_list[index] + '\n'
            f.write(line)
    else:
        f = open(save_path,mode='w+',encoding='utf-8')
        for index in range(len(image_list)):
            line = image_list[index].strip() + '\n'
            f.write(line)
