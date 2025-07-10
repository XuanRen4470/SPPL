import sys
import os

def record_accuracy(recorder_folder_path, record_file_name, accuracy):
    if not os.path.exists(recorder_folder_path):
        os.makedirs(recorder_folder_path, exist_ok=True)
        print(f"Directory '{recorder_folder_path}' created.")
    else:
        print(f"Directory '{recorder_folder_path}' already exists.")

    record_file_path = os.path.join(recorder_folder_path, record_file_name)
    with open(record_file_path, 'w') as f:
        f.write(f'{accuracy}')





def fmt_acc_to_latex(acc: float) -> str:
    """
    把 accuracy（或差值）转成带颜色的 LaTeX 文本：
      • 正数：前面加 “+”，绿色 (ForestGreen)
      • 负数：红色 (BrickRed)
      • 0   ：黑色，直接 0.00
    返回形如  \\textcolor{ForestGreen}{+1.23} 的字符串
    """
    value = float(acc * 100)         # 按你的需求 ×100 然后保留两位
    if value > 0:
        color, sign = "ForestGreen", "+"
    elif value < 0:
        color, sign = "red", ""  # 负号自带
    else:
        return f"{value:.2f}"         # 0.00 用默认颜色
    return f"\\textcolor{{{color}}}{{{sign}{value:.2f}}}"

def fmt_pho_to_latex(acc: float) -> str:
    """
    把 accuracy（或差值）转成带颜色的 LaTeX 文本：
      • 正数：前面加 “+”，绿色 (ForestGreen)
      • 负数：红色 (BrickRed)
      • 0   ：黑色，直接 0.00
    返回形如  \\textcolor{ForestGreen}{+1.23} 的字符串
    """
    value = float(acc)         # 按你的需求 ×100 然后保留两位
    if value > 0:
        color, sign = "ForestGreen", "+"
    elif value < 0:
        color, sign = "red", ""  # 负号自带
    else:
        return f"{value:.3f}"         # 0.00 用默认颜色
    return f"\\textcolor{{{color}}}{{{sign}{value:.3f}}}"




import os
from config.config import HOME_DIRECTORY

def write_to_table(experimental_result_list, table_tex_name):

    output_file = os.path.join(
        HOME_DIRECTORY, "log_total/experiment_data_recorder/latex_table", f"{table_tex_name}.tex"
    )

    # 1. 提取模型和任务列表（任务顺序固定）
    model_name_list = list(experimental_result_list[0].keys())
    # 假设每个 model 的任务集相同，取最后一个 model 的任务列表作为列名
    column_name_list = list(experimental_result_list[0][model_name_list[-1]].keys())

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 2. 写表头
    with open(output_file, "w") as f:
        f.write("\\begin{table*}[t!]\n")
        f.write("  \\centering\n")
        f.write("  \\resizebox{1.0\\textwidth}{!}{\n")
        # l|l|ccc... 格式
        col_format = "l|l|" + "|".join(["c"] * len(column_name_list))
        f.write(f"  \\begin{{tabular}}{{{col_format}}}\n")
        f.write("    \\hline\n")
        # 表头行
        header = "Methods & " + " & ".join(column_name_list)
        header = header.replace("_", " ")
        f.write(f"    {header} \\\\ \\hline\n")

        for experimental_result in experimental_result_list:
            f.write("    \\hline\n")
            # 3. 写每一行
            for i, model_name in enumerate(model_name_list):
                method = model_name.replace("_", " ")
                # 第一列 Method，第二列 Model（留空则仅写 &）
                line = f"    {method} "
                # 按固定任务顺序拼接 accuracy
                for task in column_name_list:
                    acc = experimental_result[model_name].get(task, None)
                    if acc is None:
                        line += " & "
                    else:
                        # 保留三位小数
                        if acc == '':
                            line += f" &  "
                        else:
                            if model_name.lower() == 'ours - claude' or model_name.lower() == 'ours - perplexity' or model_name.lower() == 'ours - avg of others' or model_name.lower() == 'ours filter removed - claude' or model_name.lower() == 'ours filter removed - perplexity' or model_name.lower() == 'ours filter removed - avg of others':       #'Ours Filter Removed'
                                if task.lower() == 'weighted spearman pho':
                                    line +=  f' & {fmt_pho_to_latex(acc)}' 
                                else:
                                    line += f" & " + fmt_acc_to_latex(acc) + '\%'
                            elif task.lower() == 'num of recorded data':
                                line +=  f' & {acc}' 
                            elif task.lower() == 'weighted spearman pho':
                                line +=  f' & {float(acc):.3f}' 
                            else:
                                try:
                                    line += f" & {float(acc * 100):.2f}\%"
                                except:
                                    line += f" & {acc}"
                line += " \\\\\n"
                f.write(line)

                # 行间分隔
                if 'Ours Filter Removed - Perplexity' in model_name_list:
                    if 'ours filter removed' in model_name.lower():
                        f.write("    \\hline\n")
                else:
                    if model_name.lower() == 'ours':
                        if i < len(model_name_list) - 1:
                            f.write("    \\hline\n")

        # 4. 收尾
        f.write("    \\hline\n")
        f.write("  \\end{tabular}}\n")
        f.write(f"  \\caption{{seed {table_tex_name}}}\n")
        f.write(f"  \\label{{tab:{table_tex_name}}}\n")
        f.write("\\end{table*}\n")




def write_to_table_comparison(experimental_result_list, table_tex_name):

    output_file = os.path.join(
        HOME_DIRECTORY, "log_total/experiment_data_recorder/latex_table", f"{table_tex_name}.tex"
    )

    # 1. 提取模型和任务列表（任务顺序固定）
    model_name_list = list(experimental_result_list[0].keys())
    # 假设每个 model 的任务集相同，取最后一个 model 的任务列表作为列名
    column_name_list = list(experimental_result_list[0][model_name_list[-1]].keys())

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 2. 写表头
    with open(output_file, "w") as f:
        f.write("\\begin{table*}[t!]\n")
        f.write("  \\centering\n")
        f.write("  \\resizebox{1.0\\textwidth}{!}{\n")
        # l|l|ccc... 格式
        col_format = "l|l|" + "|".join(["c"] * len(column_name_list))
        f.write(f"  \\begin{{tabular}}{{{col_format}}}\n")
        f.write("    \\hline\n")
        # 表头行
        header = "Methods & " + " & ".join(column_name_list)
        header = header.replace("_", " ")
        f.write(f"    {header} \\\\ \\hline\n")

        for experimental_result in experimental_result_list:
            f.write("    \\hline\n")
            # 3. 写每一行

            model_name_list = ['Ours - Perplexity', 'Ours Filter Removed - Perplexity']
            for i, model_name in enumerate(model_name_list):
                method = model_name.replace("_", " ")
                # 第一列 Method，第二列 Model（留空则仅写 &）
                line = f"    {method} "
                # 按固定任务顺序拼接 accuracy
                for task in column_name_list:
                    acc = experimental_result[model_name].get(task, None)
                    if acc is None:
                        line += " & "
                    else:
                        # 保留三位小数
                        if acc == '':
                            line += f" &  "
                        else:
                            if '-' in model_name:
                                if model_name.lower() == 'ours - claude' or model_name.lower() == 'ours - perplexity' or model_name.lower() == 'ours - avg of others' or model_name.lower() == 'ours filter removed - claude' or model_name.lower() == 'ours filter removed - perplexity' or model_name.lower() == 'ours filter removed - avg of others':       #'Ours Filter Removed'
                                    if task.lower() == 'weighted spearman pho':
                                        line +=  f' & {fmt_pho_to_latex(acc)}' 
                                    else:
                                        line += f" & " + fmt_acc_to_latex(acc) + '\%'
                                elif task.lower() == 'num of recorded data':
                                    line +=  f' & {acc}' 
                                elif task.lower() == 'weighted spearman pho':
                                    line +=  f' & {float(acc):.3f}' 
                                else:
                                    try:
                                        line += f" & {float(acc * 100):.2f}\%"
                                    except:
                                        line += f" & {acc}"
                line += " \\\\\n"
                f.write(line)

                # 行间分隔
                if 'Ours Filter Removed - Perplexity' in model_name_list:
                    if 'ours filter removed' in model_name.lower():
                        f.write("    \\hline\n")
                else:
                    if model_name.lower() == 'ours':
                        if i < len(model_name_list) - 1:
                            f.write("    \\hline\n")

        # 4. 收尾
        f.write("    \\hline\n")
        f.write("  \\end{tabular}}\n")
        f.write(f"  \\caption{{seed {table_tex_name}}}\n")
        f.write(f"  \\label{{tab:{table_tex_name}}}\n")
        f.write("\\end{table*}\n")
