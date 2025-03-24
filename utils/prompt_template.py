#====================
#Formatting Prompts
formatting = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''

# for mcq force answer
formatting_force = '''Respond only with the following format, nothing else:
Answer: [Repeat the ground truth answer index here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''

formatting_mcq_abstain = '''Respond only with the following format, nothing else:
Answer: [Provide answer here or 'none of the above']
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''

formatting_abstain = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here or 'unsure']
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''
#====================
#Gradual Prompts
feedback_prompt = "Env: The previous response was insufficient; explore a new line of reasoning to approach a more accurate answer, building on past information if useful."

exact_wrong_prompt = "Env: You are PROHIBITED from giving the same incorrect response as above. You MUST change your response. Give the same answer does not allow you to approach the accurate answer. If you are so sure with your answer, try rephrasing it. Ultimately, Explore a different line of reasoning or phrasing to provide a more accurate answer."

final_prompt = "Env: The response is sufficient. Stop generating further outputs."

unable_final_prompt = "Env: LLM was unable to give a sufficient response. The system has terminated."

no_answer_prompt = "Env: No answer was provided; you MUST give a response that is not none or unsure. Try exploring a new reasoning path to approach a clear and thoughtful response."

#OEQ
force_answer_prompt_gradual = (
    "You are a highly knowledgeable assistant skilled in multi-step reasoning for question answering."
    " For each question, identify key concepts, outline a logical path, and proceed with step-by-step deductions. Ensure each reasoning step clearly supports a concise, accurate answer. "
) + formatting

#MCQ
mcq_force_answer_prompt_gradual = (
    "You are a highly knowledgeable assistant skilled in multi-step reasoning for multiple-choice question answering. "
    "Based on the question and the list of choices provided, select the best answer. Carefully evaluate each option before deciding. "
    "Provide your choice (e.g., 0, 1, 2, etc) along with a brief explanation of your reasoning."
) + formatting

#Math Few-Shot
math_force_answer_prompt_gradual = (
    "You are a knowledgeable question-answering assistant. Answer the following math "
    "question using chain of thought reasoning. Here are some examples of questions and answers. " 
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    "\nThe last sentence containing your final answer must be formatted as 'The answer is __'.")

#====================
#Exploration Prompts

#mcq
mcq_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant, specializing in multiple choice questions. "
    "Based on the question and the list of choices provided, select the best answer. Carefully evaluate each option before deciding. "
    "Provide your choice (e.g., 0, 1, 2, etc) along with a brief explanation of your reasoning."
) + formatting

abstain_mcq_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant, specializing in multiple choice questions. "
    "Based on the question and the list of choices provided, select the best answer. Carefully evaluate each option before deciding. "
    "Provide your choice (e.g., 0, 1, 2, etc) or 'none of the above' along with a brief explanation of your reasoning."
) + formatting_mcq_abstain

mcq_force_rationale_prompt = (
    "You are a knowledgeable reasoner. Based on the context and the ground truth answer, explain the rationale for that answer. "
    "You should first repeat the ground truth answer and then provide a brief explanation why that is the correct answer. "
    "Your explanation should not mention any previous trial or rationale."
) + formatting_force

mcq_exploration_force_answer_prompt = (
    "You are an expert assistant specializing in multiple-choice questions, dedicated to exploring multiple ways of thinking to provide accurate answers. "
    "Below, you will see an LLM's previous answer, including the choice it selected and its reasoning, followed by the feedback: 'Wrong answer! Try again.' "
    "Even if the previous response seemed close, the answer was incorrect. Your task is to **hink outside the box** and use a **completely different line of reasoning** to approach the question. "
    "Carefully reassess each option, explore alternative interpretations of the question and choices, and **avoid repeating the same ideas or patterns** as before. "
    "Focus on providing fresh insights, considering less obvious connections, and explaining your new reasoning in a distinct way. "
    "Ensure your final choice and explanation reflect this new approach."
) + formatting


#math
math_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Answer the following math "
    "question using chain of thought reasoning. Here are some examples of questions and answers. " 
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    "\nThe last sentence containing your final answer must be formatted as 'The answer is __'.")

abstain_math_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Answer the following math "
    "question using chain of thought reasoning. Here are some examples of questions and answers. " 
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    "\nThe last sentence containing your final answer must be formatted as 'The answer is __'. If you are unsure, "
    "respond with 'unsure'.")

math_exploration_force_answer_prompt = (
    "You are an expert assistant dedicated to exploring multiple ways of thinking to provide accurate answers."
    "Answer the following math question using chain of thought reasoning. Here are some examples of questions and answers. " 

    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."

    "In addition, you will see an LLM's previous answer and its reasoning, followed by the feedback: 'Wrong answer! Try again.'"
    "Even if the previous response seemed close, the answer was incorrect."
    
    "Your task is to **think outside the box** and use a **completely different line of reasoning** to approach the question. "
    "Avoid repeating the same ideas or patterns as before. Instead, employ a new strategy, consider alternative interpretations,"
    " or explore less obvious aspects of the question. Focus on generating fresh insights and explanations."
    "Ensure your new reasoning is **distinct** from the previous one and offers a unique approach to the question."

    "The last sentence containing your final answer must be formatted as 'The answer is __'."
) 

#oeq

force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Answer the following "
    "question, and provide a brief explanation of your reasoning."
) + formatting

abstain_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Answer the following "
    "question, and provide a brief explanation of your reasoning. If you are unsure, "
    "respond with 'unsure'."
) + formatting_abstain

def create_force_rationale_prompt(answer, question_type):
    if question_type == "MATH":
        return (
            "You are a knowledgeable reasoner. Based on the context and the correct answer to a math question, explain "
            "the chain of thought reasoning behind the correct answer. Here are some examples of questions, their chain of thought reasonings, and answers. " 
            
            "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
        
            "You should provide a brief explanation why the given answer is the correct answer and then repeat the correct answer in your last sentence. "
            "Your explanation should not mention any previous trial or rationale.\n"
        
            f"The last sentence containing your final answer must be formatted as 'The answer is {answer}'."
        )
    else:
        return (
            "You are a knowledgeable reasoner. Based on the context and the ground truth answer, explain the rationale for that answer. "
            "You should first repeat the ground truth answer exactly as it is given below and then provide a brief explanation why that is the correct answer. "
            "Your explanation should not mention any previous trial or rationale.\n\n"
            f"Respond only with the following format, nothing else:\n"
            f"Answer: {answer}\n"
            "Rationale: [Provide the rationale here]\n\n"
            "Do not include any additional text, headers, or explanations outside this format."
        )


exploration_force_answer_prompt = (
    "You are an expert assistant dedicated to exploring multiple ways of thinking to provide accurate answers. Below, you will see an LLM's previous answer and its reasoning, followed by the feedback: 'Wrong answer! Try again.' Even if the previous response seemed close, the answer was incorrect. Your task is to **think outside the box** and use a **completely different line of reasoning** to approach the question. "
    "Avoid repeating the same ideas or patterns as before. Instead, employ a new strategy, consider alternative interpretations, or explore less obvious aspects of the question. Focus on generating fresh insights and explanations. Ensure your new reasoning is **distinct** from the previous one and offers a unique approach to the question."
) + formatting

