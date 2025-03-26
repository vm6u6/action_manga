import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def create_random_mask(height=512, width=512, mask_type='shapes', save_path=None):
    """
    创建随机掩码
    
    参数:
        height: 掩码高度
        width: 掩码宽度
        mask_type: 掩码类型 - 'shapes', 'noise', 'brush', 或 'blocks'
        save_path: 保存路径，如果不指定则不保存
        
    返回:
        numpy数组形式的掩码 (白色区域为要修复的部分)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if mask_type == 'shapes':
        # 创建1-5个随机形状
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            # 随机选择形状类型: 圆形、椭圆形、矩形
            shape_type = random.choice(['circle', 'ellipse', 'rectangle'])
            
            # 随机位置
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            
            if shape_type == 'circle':
                radius = random.randint(20, min(width, height) // 4)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            elif shape_type == 'ellipse':
                axis1 = random.randint(20, width // 4)
                axis2 = random.randint(20, height // 4)
                angle = random.randint(0, 180)
                cv2.ellipse(mask, (center_x, center_y), (axis1, axis2), angle, 0, 360, 255, -1)
            
            elif shape_type == 'rectangle':
                rect_width = random.randint(20, width // 3)
                rect_height = random.randint(20, height // 3)
                x1 = max(0, center_x - rect_width // 2)
                y1 = max(0, center_y - rect_height // 2)
                x2 = min(width, center_x + rect_width // 2)
                y2 = min(height, center_y + rect_height // 2)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
    elif mask_type == 'noise':
        # 创建噪声并应用阈值
        noise = np.random.rand(height, width)
        threshold = random.uniform(0.5, 0.9)  # 随机阈值
        mask[noise > threshold] = 255
        
        # 应用形态学运算使噪声区域更连贯
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
    elif mask_type == 'brush':
        # 模拟刷子笔触
        num_strokes = random.randint(3, 10)
        for _ in range(num_strokes):
            # 创建随机起点和终点
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            
            # 创建5-10个控制点来模拟一个笔画
            points = [(start_x, start_y)]
            for _ in range(random.randint(5, 10)):
                # 添加一个偏移到前一个点
                prev_x, prev_y = points[-1]
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                points.append((max(0, min(width-1, prev_x + dx)), 
                              max(0, min(height-1, prev_y + dy))))
            
            # 随机笔刷宽度
            brush_width = random.randint(5, 30)
            
            # 绘制笔画
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 255, brush_width)
                
    elif mask_type == 'blocks':
        # 创建网格块掩码
        block_size = random.randint(20, 80)
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 随机决定是否填充这个块
                if random.random() > 0.7:  # 30%的块被选为掩码
                    y2 = min(y + block_size, height)
                    x2 = min(x + block_size, width)
                    cv2.rectangle(mask, (x, y), (x2, y2), 255, -1)
    
    # 可选择保存掩码
    if save_path:
        cv2.imwrite(save_path, mask)
    
    return mask

# 创建多个掩码示例并显示
def show_mask_examples():
    plt.figure(figsize=(15, 10))
    
    mask_types = ['shapes', 'noise', 'brush', 'blocks']
    for i, mask_type in enumerate(mask_types):
        mask = create_random_mask(512, 512, mask_type)
        plt.subplot(2, 2, i+1)
        plt.title(f'{mask_type.capitalize()} Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 生成单个掩码并保存
def generate_single_mask(height=512, width=512, mask_type='shapes', save_path='random_mask.png'):
    mask = create_random_mask(height, width, mask_type, save_path)
    
    # 显示生成的掩码
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'生成的{mask_type}类型掩码')
    plt.axis('off')
    plt.show()
    
    print(f"掩码已保存到: {save_path}")
    return mask

# 示例用法
if __name__ == "__main__":
    # 显示不同类型的掩码示例
    show_mask_examples()
    
    # 生成并保存一个特定类型的掩码
    # 可选类型: 'shapes', 'noise', 'brush', 'blocks'
    generate_single_mask(mask_type='shapes', save_path='random_mask.png')