"""
AIHive Tutor AI Logic
Handles both real AI (OpenAI) and demo mode (scripted)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihive_coach.config import config
from tutor.models import StudySession
from typing import Tuple, List


class TutorAI:
    """Tutor AI that adapts based on configuration"""
    
    def __init__(self):
        self.is_demo = config.is_demo_mode()
        if not self.is_demo:
            try:
                import openai
                self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            except ImportError:
                self.is_demo = True
    
    def start_lesson(self, session: StudySession) -> str:
        """Generate the first explanation chunk and question"""
        if self.is_demo:
            return self._demo_start_lesson(session)
        else:
            return self._ai_start_lesson(session)
    
    def process_answer(self, session: StudySession, user_answer: str) -> Tuple[bool, str]:
        """
        Process user's answer to a check-for-understanding question.
        Returns: (is_correct, feedback_message)
        """
        if self.is_demo:
            return self._demo_process_answer(session, user_answer)
        else:
            return self._ai_process_answer(session, user_answer)
    
    def continue_lesson(self, session: StudySession, understood: bool) -> str:
        """
        Continue the lesson after user feedback.
        If understood=True: move to next chunk
        If understood=False: re-explain more simply
        """
        if self.is_demo:
            return self._demo_continue_lesson(session, understood)
        else:
            return self._ai_continue_lesson(session, understood)
    
    def generate_summary(self, session: StudySession) -> Tuple[str, List[str]]:
        """
        Generate end-of-session summary and 3 practice questions.
        Returns: (summary_text, [question1, question2, question3])
        """
        if self.is_demo:
            return self._demo_generate_summary(session)
        else:
            return self._ai_generate_summary(session)
    
    # Demo Mode Implementations
    
    def _demo_start_lesson(self, session: StudySession) -> str:
        """Demo mode: scripted first explanation"""
        topic = session.topic.lower()
        level = session.level
        
        # Scripted explanations for demo
        explanations = {
            "photosynthesis": {
                "Beginner": "Photosynthesis is how plants make their own food using sunlight! 🌱☀️\n\nThink of a plant like a tiny factory. The leaves are like solar panels that catch sunlight. The plant also takes in water from its roots and carbon dioxide from the air. When sunlight hits the leaves, it powers a special process that turns water and CO2 into glucose (sugar) - that's the plant's food! As a bonus, the plant releases oxygen, which we breathe.\n\n**Check your understanding:** Why does a plant need sunlight? What happens if it doesn't get enough light?",
                "Intermediate": "Photosynthesis is the biochemical process where plants convert light energy into chemical energy stored in glucose.\n\nThe process occurs in chloroplasts and involves two stages: the light-dependent reactions (in the thylakoid membrane) and the Calvin cycle (in the stroma). Light energy splits water molecules, releasing oxygen and creating ATP and NADPH. These energy carriers then power the Calvin cycle, which fixes CO2 into glucose using the enzyme RuBisCO.\n\n**Check your understanding:** Where specifically in the chloroplast do the light-dependent reactions occur, and what are their main products?",
                "Advanced": "Photosynthesis represents a complex series of redox reactions converting electromagnetic radiation into chemical potential energy. The light reactions involve photosystems II and I in a Z-scheme electron transport chain, generating a proton gradient that drives ATP synthase while reducing NADP+. The Calvin-Benson cycle operates independently of light, utilizing RuBisCO to catalyze carbon fixation through carboxylation of ribulose-1,5-bisphosphate.\n\n**Check your understanding:** Explain why the quantum yield of photosynthesis is less than 100% and identify the primary efficiency bottlenecks in C3 plants."
            },
            "python": {
                "Beginner": "Python is a programming language that lets you give instructions to a computer! 🐍💻\n\nImagine you're teaching a robot to make a sandwich. You can't just say 'make a sandwich' - you need to break it down: 'pick up bread', 'spread peanut butter', 'add jelly', etc. That's what programming is! Python is a popular language because its instructions look almost like English, making it easier to learn.\n\nHere's your first Python code:\n```python\nprint('Hello World!')\n```\nThis tells the computer to display the text 'Hello World!' on the screen.\n\n**Check your understanding:** What do you think this code does?\n```python\nprint('My name is Alex')\n```",
                "Intermediate": "Python is a high-level, interpreted programming language emphasizing code readability and simplicity.\n\nKey features include dynamic typing, automatic memory management, and extensive standard libraries. Python uses indentation for code blocks rather than braces, making structure visually clear. The language supports multiple programming paradigms: procedural, object-oriented, and functional.\n\nExample of list comprehension:\n```python\nsquares = [x**2 for x in range(10)]\n```\n\n**Check your understanding:** Why is Python called 'interpreted' rather than 'compiled', and how does this affect execution speed?",
                "Advanced": "Python implements a reference-counted garbage collection system with cycle detection, operating on CPython's bytecode VM architecture. The Global Interpreter Lock (GIL) serializes thread execution, limiting true parallelism but simplifying memory management.\n\nThe descriptor protocol enables powerful metaprogramming:\n```python\nclass LazyProperty:\n    def __get__(self, obj, objtype=None):\n        if obj is None:\n            return self\n        return obj.__dict__.setdefault(self.func.__name__, self.func(obj))\n```\n\n**Check your understanding:** How does the descriptor protocol enable Python's @property decorator, and why is __get__ called on attribute access?"
            }
        }
        
        # Default fallback for any topic
        default = {
            "Beginner": f"Let's learn about {session.topic}! This is an exciting topic. We'll start with the basics and build up your understanding step by step.\n\n**What is {session.topic}?** Think of it as... [In a real lesson, I would explain this clearly!]\n\n**Check your understanding:** Can you tell me in your own words what you think {session.topic} means?",
            "Intermediate": f"Welcome to learning about {session.topic}. At the {level} level, we'll explore the core concepts and their applications.\n\nFirst, let's establish the foundation: {session.topic} involves... [Real lesson would have detailed explanation]\n\n**Check your understanding:** How would you explain the connection between this concept and its practical applications?",
            "Advanced": f"Advanced study of {session.topic} requires deep analysis. We'll examine theoretical frameworks, edge cases, and nuanced interpretations.\n\nConsider the fundamental principles: [Real lesson would include technical depth]\n\n**Check your understanding:** Analyze the implications of these principles in complex scenarios."
        }
        
        # Look up topic in demo content
        topic_key = None
        for key in explanations.keys():
            if key in topic:
                topic_key = key
                break
        
        if topic_key:
            return explanations[topic_key][level]
        else:
            return default[level]
    
    def _demo_process_answer(self, session: StudySession, user_answer: str) -> Tuple[bool, str]:
        """Demo mode: simple keyword-based answer evaluation"""
        answer_lower = user_answer.lower()
        
        # Very simple heuristic for demo
        # In real demo, we'd have more sophisticated logic
        if len(answer_lower) < 10:
            return (False, "That's a bit brief. Can you explain with a bit more detail?")
        
        # Check for some engagement indicators
        negative_words = ["don't know", "not sure", "no idea", "confused", "help"]
        if any(word in answer_lower for word in negative_words):
            return (False, "It sounds like this part is tricky! No worries - let me explain it differently.")
        
        # Assume decent attempt
        return (True, "Great answer! You're showing good understanding. Let's move forward!")
    
    def _demo_continue_lesson(self, session: StudySession, understood: bool) -> str:
        """Demo mode: continue or simplify explanation"""
        topic = session.topic.lower()
        
        if not understood:
            # Simplify
            return f"Let me explain {session.topic} in a simpler way:\n\nThink of it like this... [Simpler explanation]. Does this make more sense now?\n\n**Let's try again:** Can you explain back to me what you understood?"
        
        # Move to next chunk
        if session.chunks_completed >= 2:
            # End lesson
            return "🎉 Excellent work! You've completed this lesson. Let me create a summary for you..."
        
        # Continue with next chunk
        return f"Perfect! Now let's explore the next part of {session.topic}...\n\n[Next explanation chunk would go here]\n\n**Check your understanding:** [Next question]"
    
    def _demo_generate_summary(self, session: StudySession) -> Tuple[str, List[str]]:
        """Demo mode: generate summary and practice questions"""
        summary = f"""**What you learned about {session.topic}:**

You explored the fundamentals of {session.topic} at the {session.level} level. Through our conversation, you:
- Understood the core concepts
- Answered {session.questions_answered_correctly} questions correctly
- Demonstrated engagement with the material

Great job on completing this study session! 🎉"""

        questions = [
            f"Explain {session.topic} in your own words to someone who has never heard of it before.",
            f"What is the most important thing to remember about {session.topic}?",
            f"How would you apply {session.topic} in a real-world situation?"
        ]
        
        return (summary, questions)
    
    # Real AI Implementations
    
    def _ai_start_lesson(self, session: StudySession) -> str:
        """Real AI mode: use OpenAI to generate first explanation"""
        system_prompt = f"""You are AIHive Tutor, a patient and encouraging study tutor.

The learner wants to study: {session.topic}
Their level: {session.level}

Your teaching style:
- Explain in short, digestible chunks (3-4 paragraphs max)
- Use clear language appropriate for {session.level} level
- End EVERY response with a check-for-understanding question
- Be encouraging and supportive
- Use analogies when helpful

Start the first lesson chunk now. Explain the basics and ask ONE specific question to check understanding."""

        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "I'll help you learn! Let's start..."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _ai_process_answer(self, session: StudySession, user_answer: str) -> Tuple[bool, str]:
        """Real AI mode: evaluate answer with OpenAI"""
        # Build conversation history
        messages = [
            {"role": "system", "content": f"You are evaluating a learner's answer about {session.topic} ({session.level} level). Determine if they understood the concept. Respond with 'CORRECT: [encouraging feedback]' or 'INCORRECT: [gentle correction]'."}
        ]
        
        # Add recent context
        for msg in session.messages[-4:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_answer})
        
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.startswith("CORRECT"):
            feedback = result.replace("CORRECT:", "").strip()
            return (True, feedback)
        else:
            feedback = result.replace("INCORRECT:", "").strip()
            return (False, feedback)
    
    def _ai_continue_lesson(self, session: StudySession, understood: bool) -> str:
        """Real AI mode: continue or simplify with OpenAI"""
        # Build conversation history
        messages = [
            {"role": "system", "content": f"""You are AIHive Tutor teaching {session.topic} ({session.level} level).
            
{'The learner understood the last concept. Continue to the next chunk.' if understood else 'The learner struggled. Re-explain the same concept more simply with a different analogy.'}

Keep it short (3-4 paragraphs). End with ONE check-for-understanding question.

If you've covered 3-4 chunks already, suggest wrapping up the lesson."""}
        ]
        
        for msg in session.messages[-6:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _ai_generate_summary(self, session: StudySession) -> Tuple[str, List[str]]:
        """Real AI mode: generate summary with OpenAI"""
        # Summarize the session
        conversation_text = "\n".join([f"{m.role}: {m.content[:100]}" for m in session.messages[-10:]])
        
        summary_response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": f"Summarize what the learner covered in this study session about {session.topic}. Be specific about key concepts. Keep it to 3-4 sentences."},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.6,
            max_tokens=300
        )
        
        summary = summary_response.choices[0].message.content.strip()
        
        # Generate practice questions
        questions_response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": f"Generate exactly 3 practice questions about {session.topic} ({session.level} level) based on what was covered. Format as a simple numbered list."},
                {"role": "user", "content": f"Session covered: {summary}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        questions_text = questions_response.choices[0].message.content.strip()
        # Parse questions
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                q = line.lstrip('0123456789.-) ').strip()
                if q:
                    questions.append(q)
        
        # Ensure exactly 3 questions
        if len(questions) < 3:
            questions.extend([
                f"Explain {session.topic} in your own words.",
                f"What is one real-world application of {session.topic}?",
                f"What question do you still have about {session.topic}?"
            ])
        
        return (summary, questions[:3])


# Global tutor AI instance
tutor_ai = TutorAI()
