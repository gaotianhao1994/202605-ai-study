from langchain.tools import Tool
import requests
import json

def web_search(query):
    try:
        search_url = "https://api.serply.io/v1/search"
        params = {
            "q": query,
            "num": 5
        }
        headers = {
            "Accept": "application/json"
        }
        
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json()
            if "results" in results:
                summaries = []
                for i, result in enumerate(results["results"][:3], 1):
                    title = result.get("title", "无标题")
                    snippet = result.get("snippet", "无摘要")
                    url = result.get("link", "无链接")
                    summaries.append(f"{i}. {title}\n   {snippet}\n   链接: {url}")
                return "\n\n".join(summaries)
            else:
                return "搜索结果为空"
        else:
            return f"搜索失败，状态码: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"网络请求失败: {str(e)}"
    except Exception as e:
        return f"搜索错误: {str(e)}"

web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="用于在互联网上搜索信息。输入应该是搜索关键词或问题。当你需要获取最新信息或不了解的知识时使用此工具。"
)

if __name__ == "__main__":
    print(web_search_tool.run("LangChain Agent 教程"))