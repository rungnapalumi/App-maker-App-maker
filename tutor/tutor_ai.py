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
            },
            "algebra": {
                "Beginner": "Algebra is a branch of mathematics where we use letters to represent numbers we don't know yet! 📐\n\nThink of algebra like a detective game. Someone tells you a clue: 'I'm thinking of a number. If you add 5 to it, you get 12. What's my number?' In algebra, we write this as: x + 5 = 12. The 'x' is our mystery number. By solving the puzzle (subtracting 5 from both sides), we find that x = 7!\n\nAlgebra is super useful because it helps us solve real problems. Want to figure out how much money you'll have after saving for 3 months? Algebra can help! Need to calculate how much pizza to order for a party? Algebra again!\n\n**Check your understanding:** If x + 3 = 10, what do you think x equals? How did you figure it out?",
                "Intermediate": "Algebra is the study of mathematical symbols and rules for manipulating these symbols. It's a powerful tool for modeling and solving real-world problems.\n\nKey concepts include variables (symbols representing unknown or changing values), expressions (combinations of variables and constants), equations (statements that two expressions are equal), and functions (relationships between inputs and outputs).\n\nFor example, the equation of a line y = mx + b uses variables: 'm' is the slope and 'b' is the y-intercept. This single equation can describe infinitely many different lines depending on the values of m and b.\n\n**Check your understanding:** In the equation 2x + 3 = 11, what steps would you take to solve for x?",
                "Advanced": "Algebra extends beyond elementary equation solving to encompass abstract algebraic structures: groups, rings, fields, vector spaces, and their morphisms. These structures satisfy specific axioms that enable generalized reasoning about mathematical operations.\n\nConsider a group (G, ∗) where ∗ is a binary operation satisfying closure, associativity, identity, and invertibility. The integers under addition form a group, as do non-zero reals under multiplication. This abstraction reveals deep structural similarities across seemingly disparate mathematical domains.\n\n**Check your understanding:** Why is the set of natural numbers under subtraction NOT a group, and which group axiom fails?"
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
            # Improved default without placeholders
            default_no_placeholders = {
                "Beginner": f"Let's learn about {session.topic}! This is an exciting topic, and I'll help you understand it step by step.\n\nTo start with the basics: {session.topic} is a concept that appears in many areas. It's something you can learn and understand, even if it seems complex at first. The key is to break it down into smaller pieces and tackle them one at a time.\n\nThink about what you already know that might be related to this topic. Building on what you already understand makes learning easier!\n\n**Check your understanding:** What do you already know about {session.topic}, or what have you heard about it before?",
                "Intermediate": f"Welcome to learning about {session.topic}. At the {level} level, we'll explore the core concepts and their practical applications.\n\nThis topic builds on foundational knowledge and extends it to more complex scenarios. You'll learn not just what {session.topic} is, but also how to apply it effectively. Understanding the underlying principles will help you recognize when and how to use this knowledge.\n\nThe key is to understand both the theory and the practice - knowing the concept AND being able to use it in real situations.\n\n**Check your understanding:** Based on what you know so far, why do you think {session.topic} is important to learn?",
                "Advanced": f"Advanced study of {session.topic} requires deep analysis and critical thinking. We'll examine theoretical frameworks, explore edge cases, and consider nuanced interpretations.\n\nAt this level, you should be comfortable with the fundamentals and ready to explore the complexities, limitations, and advanced applications. We'll look at how experts in the field approach {session.topic} and what current research reveals.\n\nThe goal is not just to understand {session.topic}, but to develop expertise that allows you to analyze, evaluate, and potentially contribute to the field.\n\n**Check your understanding:** What aspects of {session.topic} do you find most challenging or interesting, and why?"
            }
            return default_no_placeholders[level]
    
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
            # Provide real simpler explanations for demo topics
            if "photosynthesis" in topic:
                return "Let me explain photosynthesis in a simpler way:\n\nThink of a plant like a kitchen. The leaves are like a stove that uses sunlight as the heat source. The plant brings in water (from the roots) and air (through tiny holes in leaves). When sunlight hits the leaves, it's like turning on the stove - the plant 'cooks' the water and air together to make sugar, which is the plant's food. As a bonus, the plant releases oxygen, which is what we breathe!\n\nDoes this make more sense now?\n\n**Let's try again:** In your own words, what does a plant need to make food?"
            elif "python" in topic:
                return "Let me explain Python in a simpler way:\n\nThink of Python like writing a recipe. Instead of telling a person what to do, you're writing instructions for a computer. The computer follows your recipe exactly. For example, if you write 'print(\"Hello\")', you're telling the computer: 'Please display the word Hello on the screen.' The computer does exactly that - nothing more, nothing less.\n\nDoes this make more sense now?\n\n**Let's try again:** What do you think the computer does when it sees print(\"Goodbye\")?"
            elif "algebra" in topic:
                return "Let me explain algebra in a simpler way:\n\nThink of algebra like a mystery box. You know something is inside the box (that's the variable, like 'x'), but you don't know what it is yet. The equation gives you clues. For example, 'x + 3 = 7' is like saying: 'Something plus 3 equals 7, what is that something?' You can figure it out: if adding 3 gives you 7, then the mystery number must be 4!\n\nDoes this make more sense now?\n\n**Let's try again:** Can you solve this mystery: x + 5 = 10?"
            else:
                # Generic simpler explanation
                return f"Let me explain {session.topic} in a simpler way:\n\nThink of it step by step. Instead of looking at the whole concept at once, let's break it down into smaller pieces. First, focus on the most basic part - what it is and why it matters. Then we can build up from there.\n\nDoes this make more sense now?\n\n**Let's try again:** Can you tell me what part of {session.topic} makes sense to you so far?"
        
        # Move to next chunk
        if session.chunks_completed >= 2:
            # End lesson
            return "🎉 Excellent work! You've completed this lesson. Let me create a summary for you..."
        
        # Continue with next chunk - provide real continuation
        if "photosynthesis" in topic:
            return "Perfect! Now let's explore why photosynthesis is important:\n\nWithout photosynthesis, there would be no oxygen in our atmosphere, and plants wouldn't have food to grow. This means no plants, no oxygen, and no food for animals! Photosynthesis is literally what keeps most life on Earth alive.\n\n**Check your understanding:** Why would animals be in trouble if plants couldn't do photosynthesis?"
        elif "python" in topic:
            return "Perfect! Now let's explore variables in Python:\n\nVariables are like labeled boxes where you can store information. For example:\n```python\nname = \"Alice\"\nage = 25\n```\nHere, 'name' is a box holding the text \"Alice\", and 'age' is a box holding the number 25. You can use these boxes later in your code!\n\n**Check your understanding:** If you write `city = \"Paris\"`, what is stored in the 'city' variable?"
        elif "algebra" in topic:
            return "Perfect! Now let's explore why we use letters in algebra:\n\nLetters (like x, y, a, b) represent unknown numbers or numbers that can change. This is powerful because one equation can work for many situations! For example, 'distance = speed × time' works whether you're walking, driving, or flying - you just plug in different numbers for speed and time.\n\n**Check your understanding:** In the formula 'area = length × width', what do the letters represent?"
        else:
            # Generic continuation
            return f"Perfect! Now let's explore another aspect of {session.topic}:\n\nWe've covered the basics. The next important thing to understand is how this applies in real situations. This concept appears in many areas, and understanding it helps you see patterns and solve problems more easily.\n\n**Check your understanding:** Can you think of a situation where {session.topic} might be useful?"
    
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
