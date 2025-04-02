import os

def print_directory_tree(root, prefix=""):
    # 仅获取目录
    dirs = [entry for entry in os.listdir(root) if os.path.isdir(os.path.join(root, entry))]
    dirs.sort()
    for index, dir_name in enumerate(dirs):
        is_last = (index == len(dirs) - 1)
        connector = "└── " if is_last else "├── "
        print(prefix + connector + dir_name + "/")
        subdir = os.path.join(root, dir_name)
        # 根据是否最后一个目录来设置前缀
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_directory_tree(subdir, new_prefix)

if __name__ == "__main__":
    # 获取当前工作目录作为项目根目录
    root_dir = os.getcwd()
    # 打印根目录名称
    print(os.path.basename(root_dir) + "/")
    print_directory_tree(root_dir)
