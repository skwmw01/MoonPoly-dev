import os
import subprocess

# Legacy helper. The original GEMV timing binary was removed from tests/cpp and
# is not built by default. Re-enable a benchmark target under benchmarks/cpp
# before using this script.
executable_path = os.environ.get("MOONPOLY_GEMV_BENCH", "")

def run_gemv_tests():
    """
    执行外部指定的 GEMV benchmark 二进制，遍历指定的 M 和 K 值。
    遍历指定的 M 和 K 值。
    M 从 1 到 128。
    K 从 16 到 4096，且 K 是 16 的倍数。
    """


    if not executable_path:
        print("错误：MOONPOLY_GEMV_BENCH 未设置。该 legacy GEMV benchmark 不再默认构建。")
        return

    # 检查可执行文件是否存在
    if not os.path.exists(executable_path):
        print(f"错误：可执行文件未找到: {executable_path}")
        print("请确保路径正确。")
        return
    
    # 检查可执行文件是否有执行权限
    if not os.access(executable_path, os.X_OK):
        print(f"错误：文件 '{executable_path}' 没有执行权限。")
        print(f"请在Linux/macOS系统中使用 'chmod +x {executable_path}' 命令添加执行权限。")
        return

    print(f"开始执行测试，可执行文件: {executable_path}")
    print("--------------------------------------------------")
    print(f"{'M':<5} {'K':<5} {'Command':<40} {'Status':<10} {'Return Code':<12}")
    print("="*80)

    total_tests = 0
    successful_tests = 0
    failed_tests = 0

    # M 从 1 到 128
    for m_val in range(1, 128 + 1):
        # K 从 16 到 4096，步长为 16 (确保 K 是 16 的倍数)
        for k_val in range(16, 4096 + 1, 16):
            total_tests += 1
            # 构建命令列表
            command = [executable_path, str(m_val), str(k_val)]
            command_str = ' '.join(command)
            
            # 打印正在执行的命令的简略信息
            print(f"{m_val:<5} {k_val:<5} {command_str:<40}", end="")

            try:
                # 执行命令
                # capture_output=True 捕获标准输出和标准错误
                # text=True 将捕获的输出解码为文本
                result = subprocess.run(command, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    print(f"{'成功':<10} {result.returncode:<12}")
                    successful_tests += 1
                    # 如果需要查看成功时的输出，取消下面这行注释
                    if result.stdout:
                        print(f"  输出:\n{result.stdout.strip()}")
                else:
                    print(f"{'失败':<10} {result.returncode:<12}")
                    failed_tests += 1
                    # 打印标准输出和标准错误（如果有）
                    if result.stdout:
                        print(f"  标准输出:\n{result.stdout.strip()}")
                    if result.stderr:
                        print(f"  标准错误:\n{result.stderr.strip()}")
                
            except FileNotFoundError:
                print(f"{'错误':<10} {'N/A':<12}")
                print(f"  错误：命令 '{executable_path}' 未找到。请检查路径。")
                failed_tests += 1
                # 如果可执行文件在循环中途丢失，可以选择停止
                # return
            except Exception as e:
                print(f"{'异常':<10} {'N/A':<12}")
                print(f"  执行命令 '{command_str}' 时发生未知异常: {e}")
                failed_tests += 1
            
            # 为了避免输出过于密集，可以取消下面这行的注释来在每次测试后添加分隔线
            # print("-" * 80)

    print("="*80)
    print("测试执行总结:")
    print(f"  总测试用例数: {total_tests}")
    print(f"  成功测试数: {successful_tests}")
    print(f"  失败测试数: {failed_tests}")
    print("--------------------------------------------------")
    print("所有测试执行完毕。")

if __name__ == "__main__":
    run_gemv_tests()
