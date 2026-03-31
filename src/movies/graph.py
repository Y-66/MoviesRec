
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
src_path = current_file.parents[1] 
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

from movies.agents.intent_analyzer import intent_analyzer
from langchain_core.messages import HumanMessage, SystemMessage

if __name__ == "__main__":

    print("程序开始运行...") # 添加一行打印来确认进入了主函数
    
    messages = [
        SystemMessage(content="You are a Intent analyzer expert"),
        HumanMessage(content="Write a haiku about spring"),
    ]
    
    # 注意：invoke 是同步阻塞的，必须 print 它的返回值才能看到输出
    result = intent_analyzer.invoke(
        {
            "messages": messages
        },
    )
    
    print("--- 运行结果 ---")
    print(result)