# _*_ coding: utf-8 _*_
# @Time : 2022/5/6 2:25 
# @Author : yc096
# @File : code_backup.py
# ---载入部分权重
# pre_dict = torch.load(pth_path, map_location='cpu')
# now_dict = model.state_dict()
# state_dict = {k: v for k, v in pre_dict.items() if k in now_dict.keys()}
# now_dict.update(state_dict)
# model.load_state_dict(now_dict)
#---显示v6_1特征
# from Utils.inference_tools import inference_tools
# tools = inference_tools()
# tools.showFeatures(init.sigmoid(), 3, 18, 'init',resize_hw=[352, 352])
# tools.showFeatures(stage1.sigmoid(), 6, 18, 'stage1',resize_hw=[352, 352])
# tools.showFeatures(stage2.sigmoid(), 12, 18, 'stage2',resize_hw=[352, 352])
# tools.showFeatures(stage3.sigmoid(), 12, 18, 'stage3',resize_hw=[352, 352])
# tools.showFeatures(stage4.sigmoid(), 12, 18, 'stage4',resize_hw=[352, 352])
# tools.showFeatures(LFA43.sigmoid(), 3, 18, 'LFA43',resize_hw=[352, 352])
# tools.showFeatures(LFA43_.sigmoid(), 3, 18, 'LFA43_',resize_hw=[352, 352])
# tools.showFeatures(LFA32.sigmoid(), 3, 18, 'LFA32',resize_hw=[352, 352])
# tools.showFeatures(LFA32_.sigmoid(), 3, 18, 'LFA32_',resize_hw=[352, 352])
# tools.showFeatures(LFA432.sigmoid(), 3, 18, 'LFA432',resize_hw=[352, 352])
# tools.showFeatures(LFA432_.sigmoid(), 3, 18, 'LFA432_',resize_hw=[352, 352])
# tools.showFeatures(DETAIL.sigmoid(), 3, 18, 'detail432_',resize_hw=[352, 352])
# tools.showFeatures(out.sigmoid(), 1, 1, 'pred')