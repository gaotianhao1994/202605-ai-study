class PlanAndExecuteAgent:
    def __init__(self):
        self.task_list = []
    
    def plan(self, user_request):
        print(f"1. User Request: {user_request}")
        print("2. Plan - Generating tasks...")
        
        if "report" in user_request.lower():
            self.task_list = [
                "收集数据",
                "分析数据",
                "撰写报告",
                "审核报告"
            ]
        elif "email" in user_request.lower():
            self.task_list = [
                "编写邮件内容",
                "添加收件人",
                "发送邮件"
            ]
        else:
            self.task_list = [
                "理解需求",
                "制定方案",
                "执行方案",
                "验证结果"
            ]
        
        print(f"Task List: {self.task_list}")
        return self.task_list
    
    def execute_task(self, task):
        print(f"Executing task: {task}")
        return f"Task '{task}' completed successfully"
    
    def run(self, user_request):
        self.plan(user_request)
        
        print("\n3. Execute Tasks:")
        results = []
        for i, task in enumerate(self.task_list, 1):
            result = self.execute_task(task)
            results.append(result)
            print(f"  {i}. {result}")
        
        print("\n4. Loop to solve task - All tasks completed!")
        return results

if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    
    user_request = input("请输入您的请求: ")
    agent.run(user_request)