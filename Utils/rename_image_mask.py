import os
import shutil  # 复制需要


def rename_image_mask(image_root, mask_root, file_name_index=1):
    # 检查指定的image和mask文件夹路径
    assert os.path.isdir(image_root), NotADirectoryError('image root is wrong!got image pth : {}'.format(image_root))
    assert os.path.isdir(mask_root), NotADirectoryError('mask root is wrong!got image pth : {}'.format(mask_root))
    # 创建新的文件夹用于存放移动后的image和mask
    new_image_root = os.path.abspath(os.path.join(image_root, '..', 'new_images'))  # 原目录同级
    new_mask_root = os.path.abspath(os.path.join(mask_root, '..', 'new_masks'))
    if not os.path.exists(new_image_root):
        os.mkdir(new_image_root)
    if not os.path.exists(new_mask_root):
        os.mkdir(new_mask_root)
    # 获取image和mask的信息列表
    image_list = os.listdir(image_root)
    mask_list = os.listdir(mask_root)
    # 检查image和masks文件夹内数量是否一致
    assert len(image_list) == len(mask_list), RuntimeError('len(image_list != mask_list),got len(image_list):{} len(mask_list):'.format(len(image_list), len(mask_list)))
    image_list = sorted(image_list)
    mask_list = sorted(mask_list)
    # for index in range(len(image_list)):  # 默认image和mask文件名要一致，并不要求格式一样
    #     assert image_list[index].split('.')[0] == mask_list[index].split('.')[0], RuntimeError(
    #         'image.filename != mask.filename,got img.fname:{} mask.fname:{}'.format(image_list[index].split('.')[0], mask_list[index].split('.')[0]))
    # 将所有image和mask拷贝到新的目录
    for index in range(len(image_list)):
        image_path = os.path.join(image_root, image_list[index])
        mask_path = os.path.join(mask_root, mask_list[index])
        shutil.copy(image_path, new_image_root)
        shutil.copy(mask_path, new_mask_root)

    # 将新目录的所有image和mask重命名
    for index in range(len(image_list)):
        # 获取image和mask在新目录中的路径
        image_path = os.path.join(new_image_root, image_list[index])
        mask_path = os.path.join(new_mask_root, mask_list[index])
        # 检查image和mask是否存在
        assert os.path.isfile(image_path), FileExistsError('image file is not found,got image file path : {}'.format(image_path))
        assert os.path.isfile(mask_path), FileExistsError('mask file is not found,got mask file path : {}'.format(mask_path))
        # 获取新文件夹根目录
        image_file_root = os.path.dirname(image_path)  # dirname==new_image/mask_root
        mask_file_root = os.path.dirname(mask_path)
        # 获取文件名
        image_file_name = os.path.basename(image_path)
        mask_file_name = os.path.basename(mask_path)
        # 获取新文件名，如果出现os.rename与已有文件重名文件，可以修改以下两行代码，将image和mask赋一个临时名，如1temp.jpg，之后再对临时名重命名一次。
        new_image_file_name = str(file_name_index) + '.' + image_file_name.split('.')[1]
        new_mask_file_name = str(file_name_index) + '.' + mask_file_name.split('.')[1]
        # 拼接重名后的文件路径
        new_image_path = os.path.join(image_file_root, new_image_file_name)
        new_mask_path = os.path.join(mask_file_root, new_mask_file_name)
        #
        os.rename(image_path, new_image_path)
        os.rename(mask_path, new_mask_path)
        #
        file_name_index += 1
