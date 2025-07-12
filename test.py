# import torch
# import timm
# import models_mamba  # 注册模型必须导入
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')
#
# # 创建模型并移动到GPU
# model = timm.create_model('test', pretrained=False).to(device)
# # model.eval()
# #
# # # 构造虚拟输入，并移动到GPU
# # batch_size = 2
# # img_size = (3, 1280, 960)
# # lidar_size = (1, 1280, 960)
# #
# # dummy_img = torch.randn(batch_size, *img_size).to(device)
# # dummy_lidar = torch.randn(batch_size, *lidar_size).to(device)
# #
# # # 前向传播
# # with torch.no_grad():
# #     output = model(dummy_img, dummy_lidar)
# #
# # # 打印输出形状
# # print("输出 keys:", output.keys())
# # print("分类 logits 形状:", output['pred_logits'].shape)
# # print("边框 boxes 形状:", output['pred_boxes'].shape)
#
# model.train()  # 切换到训练模式（启用Dropout等）
#
# # 构造虚拟输入和标签
# batch_size = 2
# img_channels = 3
# lidar_channels = 1
# H, W = 1280, 960
#
# # 生成输入数据
# dummy_img = torch.randn(batch_size, img_channels, H, W).to(device)
# dummy_lidar = torch.randn(batch_size, lidar_channels, H, W).to(device)
#
# # 生成虚拟标签（假设任务为检测100个目标）
# num_queries = 100
# num_classes = 1  # 假设二分类（背景+目标）
#
# # 分类标签（形状：[batch_size, num_queries]）
# # 随机生成0或1的类别标签（0: 背景，1: 目标）
# target_classes = torch.randint(0, num_classes+1, (batch_size, num_queries)).to(device)
#
# # 边框标签（形状：[batch_size, num_queries, 4]）
# # 随机生成归一化的坐标 [cx, cy, w, h]
# target_boxes = torch.rand(batch_size, num_queries, 4).to(device)
#
# # 前向传播
# outputs = model(dummy_img, dummy_lidar)
#
# # 定义损失函数
# def compute_loss(outputs, targets):
#     # 分类损失（交叉熵）
#     pred_logits = outputs['pred_logits'].flatten(0, 1)  # [batch*num_queries, num_classes+1]
#     target_classes = targets['classes'].flatten()       # [batch*num_queries]
#     loss_cls = torch.nn.functional.cross_entropy(pred_logits, target_classes)
#
#     # 边框损失（L1）
#     pred_boxes = outputs['pred_boxes'].flatten(0, 1)    # [batch*num_queries, 4]
#     target_boxes = targets['boxes'].flatten(0, 1)       # [batch*num_queries, 4]
#     loss_box = torch.nn.functional.l1_loss(pred_boxes, target_boxes)
#
#     # 总损失（可根据任务调整权重）
#     total_loss = loss_cls + 0.5 * loss_box
#     return total_loss
#
# # 包装标签
# targets = {
#     'classes': target_classes,
#     'boxes': target_boxes
# }
#
# # 计算损失
# loss = compute_loss(outputs, targets)
#
# # 反向传播
# model.zero_grad()  # 清空梯度
# loss.backward()    # 计算梯度
#
# # 检查梯度是否存在且有效
# has_valid_gradients = False
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         has_valid_gradients = True
#         if torch.isnan(param.grad).any():
#             print(f"梯度包含NaN值: {name}")
#         if torch.isinf(param.grad).any():
#             print(f"梯度包含Inf值: {name}")
#
# if has_valid_gradients:
#     print("反向传播成功！所有参数梯度已计算。")
# else:
#     print("错误：未检测到有效梯度。")
#
# # 打印损失值和梯度统计
# print(f"总损失值: {loss.item():.4f}")
# print("梯度范数示例（前5个参数）:")
# for name, param in list(model.named_parameters())[:5]:
#     if param.grad is not None:
#         grad_norm = param.grad.norm().item()
#         print(f"{name:30}梯度范数: {grad_norm:.6f}")
#         print(f"{name:30}梯度范数: {grad_norm:.6f}")


import torch
import torch
from models_mamba import VisionMamba



if __name__ == "__main__":


    # 模拟输入：B=1, C=3, H=960, W=1280
    dummy_img = torch.randn(1, 3, 960, 1280)
    dummy_lidar_mask = torch.randn(1, 1, 960, 1280)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionMamba(
        img_size=(1280, 960),
        patch_size=64,
        depth=24,
        embed_dim=768,
        in_chans=3,
        num_classes=1,  # 检测头只预测1类
        if_cls_token=True,
        if_abs_pos_embed=True,
        if_rope=False,  # 可以先关闭 Rope 方便调试
        if_bimamba=True,
        final_pool_type='all'
    )
    model.eval()


    model = model.to(device)
    dummy_img = dummy_img.to(device)
    dummy_lidar_mask = dummy_lidar_mask.to(device)


    with torch.no_grad():
        output = model(dummy_img, dummy_lidar_mask, return_features=True)
        print("\n✅ Forward 正常运行！")
        if isinstance(output, tuple):
            print("预测 shape:", output[0]['pred_logits'].shape)  # [B, num_queries, num_classes]
            print("fused 特征 shape:", output[1].shape)           # [B, num_tokens, C]
        else:
            print("输出 shape:", output.shape)
