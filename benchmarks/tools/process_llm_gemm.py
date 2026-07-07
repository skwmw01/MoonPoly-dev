import os

# m,n,k
# llama-3-8b
# 1,4096,4096
# 1,4096,14336
# 1,14336,4096

# llama-2-7b
# 1,4096,11008
# 1,11008, 4096

# llama-2-13B
# 5120, 13824
# 13824, 5120

def convert_string_to_csv(input_filename="llm_bench.csv", output_filename="output.csv"):
    """
    Reads a comma-separated string of 'num_num_num' items from an input file,
    converts each item to 'num,num,num', and saves them to an output CSV file,
    with each entry on a new line.
    """
    print(f"▶️  开始处理...")
    print(f"  - 输入文件: '{input_filename}'")
    print(f"  - 输出文件: '{output_filename}'")

    # 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"\n❌ 错误: 输入文件 '{input_filename}' 未找到。")
        # 创建一个示例输入文件来帮助用户
        try:
            with open(input_filename, 'w') as f:
                f.write("4096_512_11008,4096_512_1100,8192_1024_22016")
            print(f"   已为您创建一个示例 '{input_filename}' 文件。请用您的数据填充它后重试。")
        except IOError as e:
            print(f"   创建示例文件失败: {e}")
        return

    # 读取输入文件的内容
    try:
        with open(input_filename, 'r') as f:
            # 读取所有内容并移除可能的前后空白
            content = f.read().strip()
    except IOError as e:
        print(f"❌ 读取文件 '{input_filename}' 时出错: {e}")
        return

    # 如果文件是空的，则退出
    if not content:
        print("\n🤷 输入文件是空的，无需处理。")
        return

    # 按逗号分割字符串，得到所有条目
    items = content.split(',')

    # 过滤掉可能因多余逗号产生的空字符串条目
    items = [item.strip() for item in items if item.strip()]
    
    print(f"\n🔍 找到 {len(items)} 个条目需要转换。")

    # 准备写入 CSV 文件
    try:
        with open(output_filename, 'w', newline='') as f:
            print(f"✍️  正在写入 '{output_filename}'...")
            count = 0
            for item in items:
                # 检查格式是否大致正确 (可选，但推荐)
                parts = item.split('_')
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    # 将下划线替换为逗号
                    csv_line = item.replace('_', ',')
                    # 写入文件，并在末尾添加换行符
                    f.write(csv_line + '\n')
                    count += 1
                else:
                    print(f"  -  警告: 忽略格式不正确的条目 '{item}'")
            
            print(f"\n🎉 成功！共 {count} 个有效条目已写入 '{output_filename}'。")

    except IOError as e:
        print(f"\n❌ 写入文件时出错: {e}")

# --- 运行主函数 ---
if __name__ == "__main__":
    convert_string_to_csv()
