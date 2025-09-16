import sys

def add_numbers_to_headings(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 1
    new_lines = []
    for line in lines:
        if line.startswith("# "):  # 只处理一级标题
            # 去掉已有的数字序号
            content = line[2:].lstrip()
            if content[0].isdigit():
                # 如果本来有序号，去掉
                content = content.split(" ", 1)[-1]
            new_line = f"# {count} {content}"
            count += 1
        else:
            new_line = line
        new_lines.append(new_line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python add_heading_numbers.py input.md output.md")
    else:
        add_numbers_to_headings(sys.argv[1], sys.argv[2])

