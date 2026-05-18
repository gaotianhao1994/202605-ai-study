from langchain.tools import Tool
import os

def read_file(file_path):
    try:
        file_path = file_path.strip()
        if not os.path.isfile(file_path):
            return f"错误：文件 '{file_path}' 不存在"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 2000:
            content = content[:2000] + "\n\n...（文件内容过长，仅显示前2000字符）"
        
        return content
    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 未找到"
    except PermissionError:
        return f"错误：没有权限读取文件 '{file_path}'"
    except Exception as e:
        return f"读取文件错误: {str(e)}"

def write_file(file_path, content):
    try:
        file_path = file_path.strip()
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"文件 '{file_path}' 已成功写入"
    except PermissionError:
        return f"错误：没有权限写入文件 '{file_path}'"
    except Exception as e:
        return f"写入文件错误: {str(e)}"

read_file_tool = Tool(
    name="read_file",
    func=read_file,
    description="用于读取文件内容。输入应该是文件的绝对路径。当你需要查看文件内容时使用此工具。"
)

write_file_tool = Tool(
    name="write_file",
    func=write_file,
    description="用于写入内容到文件。输入应该是 '文件路径|内容' 的格式。当你需要保存内容到文件时使用此工具。"
)

if __name__ == "__main__":
    print(read_file_tool.run(__file__))
    print(write_file_tool.run("/tmp/test_write.txt|这是测试内容"))