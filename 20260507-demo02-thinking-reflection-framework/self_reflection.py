#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我反思模块
核心功能：对思考过程进行反思、评估和改进

这个模块模拟了人类的自我反思能力，包括：
1. 监控思考过程
2. 评估决策质量
3. 识别思考偏差
4. 提出改进建议
5. 从经验中学习
"""

from typing import List, Dict, Any, Optional
import time
from thinking_engine import Thought


class Reflection:
    """
    单个反思记录
    记录对一次思考过程的反思结果
    """
    
    def __init__(self, thought_history: List[Thought], outcome=None):
        """
        初始化反思记录
        
        Args:
            thought_history: 被反思的思考过程
            outcome: 实际结果（如果已知），用于对比预期与实际
        """
        self.thought_history = thought_history
        self.outcome = outcome
        self.reflection_time = time.time()
        self.assessments = []  # 评估结果列表
        self.biases = []       # 识别出的偏差列表
        self.suggestions = []  # 改进建议列表
    
    def __repr__(self):
        return f"Reflection(assessments={len(self.assessments)}, biases={len(self.biases)}, suggestions={len(self.suggestions)})"


class SelfReflectionEngine:
    """
    自我反思引擎
    负责对思考过程进行深度反思
    
    反思维度包括：
    1. 过程完整性：思考步骤是否完整
    2. 逻辑一致性：推理是否连贯
    3. 证据充分性：是否有足够的证据支持决策
    4. 偏差识别：是否存在认知偏差
    5. 结果对比：预期结果与实际结果的差异
    """
    
    def __init__(self, memory_system=None):
        """
        初始化反思引擎
        
        Args:
            memory_system: 外部记忆系统，用于存储反思结果
        """
        self.memory_system = memory_system
        self.reflection_history = []  # 历史反思记录
    
    def reflect(self, thought_history: List[Thought], outcome=None) -> Dict[str, Any]:
        """
        执行完整的反思过程
        
        Args:
            thought_history: 要反思的思考过程
            outcome: 实际结果（可选）
        
        Returns:
            反思结果字典，包含评估、偏差和改进建议
        """
        reflection = Reflection(thought_history, outcome)
        
        # 1. 评估思考过程完整性
        self._assess_completeness(reflection)
        
        # 2. 评估逻辑一致性
        self._assess_consistency(reflection)
        
        # 3. 评估证据充分性
        self._assess_evidence(reflection)
        
        # 4. 识别认知偏差
        self._identify_biases(reflection)
        
        # 5. 如果有实际结果，进行结果对比
        if outcome is not None:
            self._compare_with_outcome(reflection)
        
        # 6. 生成综合改进建议
        self._generate_suggestions(reflection)
        
        # 保存反思记录
        self.reflection_history.append(reflection)
        
        # 如果有记忆系统，保存反思结果
        if self.memory_system:
            self.memory_system.store({
                'type': 'reflection',
                'content': str(reflection),
                'timestamp': time.time(),
                'quality': self._calculate_quality_score(reflection)
            })
        
        return {
            'assessments': reflection.assessments,
            'biases': reflection.biases,
            'suggestions': reflection.suggestions,
            'overall_quality': self._calculate_quality_score(reflection)
        }
    
    def _assess_completeness(self, reflection: Reflection):
        """
        评估思考过程的完整性
        
        检查思考是否包含了必要的阶段：
        - 感知阶段
        - 分析阶段
        - 生成阶段
        - 评估阶段
        - 决策阶段
        """
        thought_types = [t.thought_type for t in reflection.thought_history]
        
        required_phases = {
            'observation': '感知阶段',
            'analysis': '分析阶段',
            'generation': '生成阶段',
            'evaluation': '评估阶段',
            'decision': '决策阶段'
        }
        
        missing_phases = []
        for phase_type, phase_name in required_phases.items():
            if phase_type not in thought_types:
                missing_phases.append(phase_name)
        
        if missing_phases:
            assessment = {
                'aspect': '完整性',
                'rating': 0.6,
                'comment': f"缺少以下阶段：{', '.join(missing_phases)}",
                'suggestion': '建议补充缺失的思考阶段'
            }
        else:
            phase_count = len(thought_types)
            if phase_count >= 8:
                assessment = {
                    'aspect': '完整性',
                    'rating': 0.9,
                    'comment': '思考过程完整，包含所有必要阶段',
                    'suggestion': None
                }
            elif phase_count >= 5:
                assessment = {
                    'aspect': '完整性',
                    'rating': 0.75,
                    'comment': '思考过程基本完整',
                    'suggestion': '可以增加更多细节分析'
                }
            else:
                assessment = {
                    'aspect': '完整性',
                    'rating': 0.5,
                    'comment': '思考过程过于简略',
                    'suggestion': '建议增加更多思考步骤'
                }
        
        reflection.assessments.append(assessment)
    
    def _assess_consistency(self, reflection: Reflection):
        """
        评估思考逻辑的一致性
        
        检查思考过程是否存在逻辑矛盾或跳跃
        """
        confidence_scores = [t.confidence for t in reflection.thought_history]
        thought_count = len(reflection.thought_history)
        
        if thought_count < 3:
            assessment = {
                'aspect': '一致性',
                'rating': 0.5,
                'comment': '思考步骤太少，无法充分评估一致性',
                'suggestion': '增加思考步骤以提高可验证性'
            }
            reflection.assessments.append(assessment)
            return
        
        # 检查置信度波动
        max_confidence = max(confidence_scores)
        min_confidence = min(confidence_scores)
        confidence_range = max_confidence - min_confidence
        
        # 检查是否有矛盾的思考
        contradictory_count = 0
        for i, thought1 in enumerate(reflection.thought_history):
            for j, thought2 in enumerate(reflection.thought_history[i+1:], i+1):
                if '不' in thought1.content and '不' not in thought2.content:
                    # 简单的矛盾检测
                    contradictory_count += 1
        
        if contradictory_count > 0:
            assessment = {
                'aspect': '一致性',
                'rating': 0.4,
                'comment': f'检测到 {contradictory_count} 处潜在矛盾',
                'suggestion': '需要重新审视相互矛盾的观点'
            }
        elif confidence_range > 0.4:
            assessment = {
                'aspect': '一致性',
                'rating': 0.65,
                'comment': f'置信度波动较大 ({min_confidence:.2f} ~ {max_confidence:.2f})',
                'suggestion': '建议对低置信度的思考进行验证'
            }
        else:
            assessment = {
                'aspect': '一致性',
                'rating': 0.85,
                'comment': '思考逻辑连贯，置信度稳定',
                'suggestion': None
            }
        
        reflection.assessments.append(assessment)
    
    def _assess_evidence(self, reflection: Reflection):
        """
        评估证据的充分性
        
        检查思考过程是否有足够的证据支持决策
        """
        evidence_keywords = ['根据', '因为', '基于', '参考', '数据', '事实', '证据']
        
        evidence_count = sum(
            1 for thought in reflection.thought_history
            if any(keyword in thought.content for keyword in evidence_keywords)
        )
        
        thought_count = len(reflection.thought_history)
        evidence_ratio = evidence_count / max(thought_count, 1)
        
        if evidence_ratio >= 0.3:
            assessment = {
                'aspect': '证据充分性',
                'rating': 0.8,
                'comment': f'证据充足，{evidence_count} 条思考有明确依据',
                'suggestion': None
            }
        elif evidence_ratio >= 0.1:
            assessment = {
                'aspect': '证据充分性',
                'rating': 0.55,
                'comment': f'证据一般，只有 {evidence_count} 条思考有明确依据',
                'suggestion': '建议增加更多事实依据和数据支持'
            }
        else:
            assessment = {
                'aspect': '证据充分性',
                'rating': 0.3,
                'comment': '缺乏证据支持，大部分思考没有明确依据',
                'suggestion': '需要寻找更多证据来支持决策'
            }
        
        reflection.assessments.append(assessment)
    
    def _identify_biases(self, reflection: Reflection):
        """
        识别潜在的认知偏差
        
        常见认知偏差类型：
        - 确认偏差：只寻找支持自己观点的证据
        - 锚定效应：过度依赖初始信息
        - 过度自信：对自己的判断过于自信
        - 从众效应：跟随他人意见
        - 损失厌恶：对损失的感受比对收益更强烈
        """
        bias_patterns = [
            {
                'name': '确认偏差',
                'keywords': ['显然', '肯定', '毫无疑问', '当然'],
                'description': '倾向于只寻找支持自己观点的证据'
            },
            {
                'name': '过度自信',
                'keywords': ['绝对', '完全', '一定', '必定'],
                'description': '对自己的判断过于自信'
            },
            {
                'name': '锚定效应',
                'keywords': ['首先', '最初', '根据之前', '基于初始'],
                'description': '过度依赖初始信息'
            },
            {
                'name': '从众效应',
                'keywords': ['大家都', '大多数人', '普遍认为', '通常'],
                'description': '倾向于跟随他人意见'
            }
        ]
        
        for bias in bias_patterns:
            for thought in reflection.thought_history:
                if any(keyword in thought.content for keyword in bias['keywords']):
                    reflection.biases.append({
                        'name': bias['name'],
                        'description': bias['description'],
                        'found_in': thought.content[:50] + '...',
                        'severity': '中等'
                    })
                    break  # 每种偏差只记录一次
        
        # 检查整体置信度是否过高
        avg_confidence = sum(t.confidence for t in reflection.thought_history) / max(len(reflection.thought_history), 1)
        if avg_confidence > 0.9:
            reflection.biases.append({
                'name': '过度自信',
                'description': '整体置信度过高，可能存在过度自信偏差',
                'found_in': f'平均置信度: {avg_confidence:.2f}',
                'severity': '高'
            })
    
    def _compare_with_outcome(self, reflection: Reflection):
        """
        对比预期结果与实际结果
        
        如果提供了实际结果，评估决策的准确性
        """
        if reflection.outcome is None:
            return
        
        # 找到决策阶段的思考
        decision_thoughts = [t for t in reflection.thought_history if t.thought_type == 'decision']
        
        if decision_thoughts:
            last_decision = decision_thoughts[-1]
            
            # 简单对比：检查结果是否包含"成功"或"失败"
            if '成功' in str(reflection.outcome) or '完成' in str(reflection.outcome):
                match = 'positive'
            elif '失败' in str(reflection.outcome) or '错误' in str(reflection.outcome):
                match = 'negative'
            else:
                match = 'neutral'
            
            assessment = {
                'aspect': '结果匹配度',
                'rating': 0.9 if match == 'positive' else 0.4 if match == 'negative' else 0.6,
                'comment': f"决策结果与实际结果{'相符' if match == 'positive' else '不符' if match == 'negative' else '无法确定'}",
                'suggestion': '继续观察后续结果以验证决策质量' if match == 'neutral' else None
            }
            reflection.assessments.append(assessment)
    
    def _generate_suggestions(self, reflection: Reflection):
        """
        根据评估结果生成综合改进建议
        """
        # 收集所有评估中的建议
        for assessment in reflection.assessments:
            if assessment.get('suggestion'):
                reflection.suggestions.append(assessment['suggestion'])
        
        # 根据偏差生成针对性建议
        for bias in reflection.biases:
            if bias['name'] == '确认偏差':
                reflection.suggestions.append('建议主动寻找反驳自己观点的证据')
            elif bias['name'] == '过度自信':
                reflection.suggestions.append('建议降低置信度，增加验证步骤')
            elif bias['name'] == '锚定效应':
                reflection.suggestions.append('建议在做决策前考虑多种可能性')
            elif bias['name'] == '从众效应':
                reflection.suggestions.append('建议独立思考，不盲目跟随他人')
        
        # 去重
        reflection.suggestions = list(set(reflection.suggestions))
    
    def _calculate_quality_score(self, reflection: Reflection) -> float:
        """
        计算思考过程的综合质量分数
        
        Returns:
            0.0-1.0 的质量分数
        """
        if not reflection.assessments:
            return 0.0
        
        total_rating = sum(a['rating'] for a in reflection.assessments)
        avg_rating = total_rating / len(reflection.assessments)
        
        # 考虑偏差数量的影响
        bias_penalty = min(len(reflection.biases) * 0.05, 0.2)
        
        return max(0.0, min(1.0, avg_rating - bias_penalty))
    
    def get_reflection_history(self) -> List[Reflection]:
        """
        获取历史反思记录
        
        Returns:
            所有反思记录的列表
        """
        return self.reflection_history


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("自我反思模块测试")
    print("=" * 60)
    
    from thinking_engine import ThinkingEngine
    
    # 创建思考引擎并生成思考过程
    engine = ThinkingEngine()
    result = engine.think("如何提高学习效率？")
    
    # 创建反思引擎并进行反思
    reflector = SelfReflectionEngine()
    reflection_result = reflector.reflect(result['thoughts'], outcome='成功')
    
    # 输出反思结果
    print("\n反思评估结果：")
    for i, assessment in enumerate(reflection_result['assessments'], 1):
        print(f"{i:2d}. [{assessment['aspect']}] 评分: {assessment['rating']:.2f}")
        print(f"    评价: {assessment['comment']}")
        if assessment['suggestion']:
            print(f"    建议: {assessment['suggestion']}")
    
    print("\n识别到的偏差：")
    for i, bias in enumerate(reflection_result['biases'], 1):
        print(f"{i:2d}. [{bias['severity']}] {bias['name']}: {bias['description']}")
    
    print("\n改进建议：")
    for i, suggestion in enumerate(reflection_result['suggestions'], 1):
        print(f"{i:2d}. {suggestion}")
    
    print(f"\n综合质量评分：{reflection_result['overall_quality']:.2f}")