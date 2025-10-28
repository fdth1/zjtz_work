import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from glm_config import ProjectConfig
import peft

pc = ProjectConfig()

class TripletExtractor:
    def __init__(self, model_path=None):
        """
        初始化三元组抽取器
        
        Args:
            model_path (str): 微调后的模型路径，如果为None则使用配置中的最佳模型路径
        """
        self.model_path = model_path or os.path.join(pc.save_dir, "model_best")
        self.device = pc.device
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        print(f"🤖 加载模型从: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if pc.fp16 else torch.float32
        ).to(self.device)
        
        self.model.eval()
        print("✅ 模型加载完成")
        
    def create_prompt(self, text: str) -> str:
        """
        创建三元组抽取的提示词
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 格式化的提示词
        """
        prompt = f"""Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。
Input: 帮我抽取出下面句子中的三元组信息，返回JSON：

{text}
Answer: """
        return prompt
        
    def extract_triplets(self, text: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        从文本中抽取三元组
        
        Args:
            text (str): 输入文本
            max_length (int): 生成的最大长度
            temperature (float): 生成温度，越低越确定性
            
        Returns:
            str: 生成的三元组JSON字符串
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        prompt = self.create_prompt(text)
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案部分
        if "Answer: " in response:
            answer = response.split("Answer: ")[-1].strip()
        else:
            answer = response
            
        return answer
    
    def parse_triplets(self, response: str) -> list:
        """
        解析生成的三元组JSON
        
        Args:
            response (str): 模型生成的响应
            
        Returns:
            list: 解析后的三元组列表
        """
        try:
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                # 如果没有代码块标记，尝试直接解析
                json_str = response.strip()
                
            # 解析JSON
            triplets = json.loads(json_str)
            return triplets if isinstance(triplets, list) else [triplets]
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始响应: {response}")
            return []
    
    def extract_and_parse(self, text: str, **kwargs) -> list:
        """
        抽取并解析三元组（一步到位）
        
        Args:
            text (str): 输入文本
            **kwargs: 传递给extract_triplets的参数
            
        Returns:
            list: 三元组列表
        """
        response = self.extract_triplets(text, **kwargs)
        return self.parse_triplets(response)


def main():
    parser = argparse.ArgumentParser(description="ChatGLM-6B 三元组抽取推理")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--text", type=str, default=None, help="要抽取三元组的文本")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    parser.add_argument("--max_length", type=int, default=512, help="最大生成长度")
    
    args = parser.parse_args()
    
    # 初始化抽取器
    extractor = TripletExtractor(args.model_path)
    
    if args.interactive:
        print("🎯 进入交互式三元组抽取模式")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 50)
        
        while True:
            text = input("\n请输入要抽取三元组的文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
                
            if not text:
                print("⚠️ 请输入有效文本")
                continue
                
            print("\n🔍 正在抽取三元组...")
            try:
                triplets = extractor.extract_and_parse(
                    text, 
                    temperature=args.temperature,
                    max_length=args.max_length
                )
                
                if triplets:
                    print(f"\n✅ 抽取到 {len(triplets)} 个三元组:")
                    for i, triplet in enumerate(triplets, 1):
                        print(f"{i}. 主体: {triplet.get('subject', 'N/A')}")
                        print(f"   关系: {triplet.get('predicate', 'N/A')}")
                        print(f"   客体: {triplet.get('object', 'N/A')}")
                        print(f"   主体类型: {triplet.get('subject_type', 'N/A')}")
                        print(f"   客体类型: {triplet.get('object_type', 'N/A')}")
                        print()
                else:
                    print("❌ 未能抽取到有效三元组")
                    
            except Exception as e:
                print(f"❌ 抽取过程中出错: {e}")
    
    elif args.text:
        print(f"🔍 抽取文本: {args.text}")
        extractor = TripletExtractor(args.model_path)
        
        try:
            triplets = extractor.extract_and_parse(
                args.text,
                temperature=args.temperature,
                max_length=args.max_length
            )
            
            print(f"\n✅ 抽取结果:")
            print(json.dumps(triplets, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f"❌ 抽取失败: {e}")
    
    else:
        # 演示模式
        print("🎯 ChatGLM-6B 三元组抽取演示")
        
        demo_texts = [
            "《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧。",
            "爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子。",
            "南迦帕尔巴特峰，8125米。"
        ]
        
        for i, text in enumerate(demo_texts, 1):
            print(f"\n📝 示例 {i}: {text}")
            try:
                triplets = extractor.extract_and_parse(text)
                print(f"✅ 抽取结果: {json.dumps(triplets, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"❌ 抽取失败: {e}")


if __name__ == "__main__":
    main()