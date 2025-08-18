from config.config import MODEL_ENGINE, GPT_API

math_quality_score_rubric = \
f"""sDoes the solution demonstrate mathemati- cal correctness, reasoning, clarity, and pre- cision?
Score 1 Description: The solution is incor- rect or mathematically flawed, with major errors in reasoning, calculations, or logic, making the answer unusable.
Score 2 Description: The solution contains relevant or partially correct information, but has significant errors in calculations or rea- soning that substantially affect the result. Score 3 Description: The solution is mostly correct but may contain minor mistakes or gaps in reasoning. The overall structure and approach are sound, but some calculations or logic may need refinement.
Score 4 Description: The solution is correct, well-reasoned, and clear, though there may be slight room for improvement or minor refinements to become a perfect solution to the problem.
Score 5 Description: The solution is excel- lent, fully correct, and demonstrates a high level of mathematical precision, clarity, and creativity, with well-articulated reasoning and no errors."""

instruction_following_response_quality_rubric = \
f"""Does the response consider a wide range of factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail?
Score 1 Description: The response is not helpful at all or seems helpful on the surface but is actually incorrect such as including in- correct information, naive miscalculations, or unexecutable code.
Score 2 Description: The response contains some relevant or helpful information, but also has major flaws interms of factuality, accuracy, and relevance.
Score 3 Description: The response is mostly correct but minor flaws regarding factuality, accuracy, and relevance still exists, while it is overall an okay response.
Score 4 Description: The response is accu- rate, relevant, and helpful, although there are some slight improvements that could be made when an expert evaluates the re- sponse.
Score 5 Description: The response is ex- cellent. It is completely factual, accurate, relevant, and helpful, demonstrating a high degree of depth and creativity."""

code_rubric = \
f"""How effective, efficient, and logically sound is the code solution, focusing on per- formance, executability, and correctness?
Score 1 Description: The code contains fun- damental logic or syntax errors, making it incorrect or unexecutable. It fails to com- plete the intended task or produces entirely incorrect outputs.
Score 2 Description: The code is partially functional but contains major logic errors or inefficiencies that significantly impact performance or correctness. It may run but produces incorrect or incomplete results. Score 3 Description: The code is mostly correct and executable, though there may be minor logic issues, inefficiencies, or sub- optimal use of data structures or algorithms. The solution functions as intended, but im- provements could make it more efficient or robust.
Score 4 Description: The code is fully cor- rect, functional, and reasonably efficient. It completes the task as intended, balancing performance with logical soundness. Mi- nor optimizations could still enhance perfor- mance.
Score 5 Description: The code is fully cor- rect, optimally efficient, and logically ro- bust, providing the best possible perfor- mance for the task. It executes flawlessly without errors or any significant room for improvement."""

instruction_following_instruction_difficulty_rubric = \
f"""How complex and challenging is the given instruction to answer perfectly?
Score 1 Description: The instruction re- quires only factual knowledge, without any need for reasoning or critical thinking. A straightforward, single-step response suf- fices.
Score 2 Description: The instruction re- quires some reasoning, such as explaining a concept involving multiple simple ideas, solving a straightforward problem, or pro- viding a response that involves a few logical steps, though still simple in nature.
Score 3 Description: The instruction re- quires a substantial amount of reasoning and the integration of multiple related concepts. Answering it accurately involves a multi- step process and may require intermediate- level knowledge or analytical thinking. Score 4 Description: The instruction re- quires advanced reasoning, demanding deep understanding of complex concepts or sub- stantial problem-solving. Answering it re- quires carefully navigating multiple interre- lated ideas or steps, often involving special- ized knowledge or sophisticated analytical skills.
Score 5 Description: The instruction is ex- ceptionally challenging and requires high- level reasoning or novel problem-solving. It involves extensive conceptual understand- ing, abstraction, and potentially innovative thinking, with substantial effort required to arrive at an accurate and complete answer."""

math_instruction_difficulty_rubric = \
f"""How complex and challenging is the math problem to solve?
Score 1 Description: The problem requires only simple operations or direct application of a single, basic concept. Minimal rea- soning is needed, and the solution follows immediately from applying a known rule or formula.
Score 2 Description: The problem requires basic reasoning and involves applying a fa- miliar formula or concept with slight varia- tion. It may involve a straightforward multi- step process, but each step is clear and relies on commonly used methods.
Score 3 Description: The problem requires moderate reasoning, combining multiple concepts that interact in a meaningful way. Solving it involves several steps and may require logical sequencing or some abstrac- tion, but the approach is approachable with a solid foundational understanding.
Score 4 Description: The problem demands advanced reasoning, involving multiple in- terdependent concepts that require careful coordination. Solution steps are less ob- vious, requiring critical thinking and pos- sibly choosing between multiple solution paths. Solving the problem involves more abstract reasoning or creative application of concepts.
Score 5 Description: The problem is ex- tremely complex and demands sophisticated reasoning and problem-solving skills. It may involve novel combinations of con- cepts, intricate logical chains, or innovative approaches to solve. This level typically requires significant abstraction, exploration of unconventional methods, and flexibility in adapting mathematical tools."""


code_instruction_difficulty_rubric = \
f"""How complex and challenging is the coding problem to solve?
Score 1 Description: The problem involves implementing simple functionality or a di- rect operation. It requires minimal logic, with a straightforward approach and no com- plex decision-making.
Score 2 Description: The problem requires basic control flow, such as using loops or conditional statements. The logic is clear and sequential, with minimal interaction be- tween different parts of the code.
Score 3 Description: The problem involves intermediate logic, combining multiple pro- gramming constructs and requiring a coher- ent structure. Solving it requires handling a sequence of steps with basic data manip- ulation, but follows a familiar, manageable approach.
Score 4 Description: The problem demands advanced reasoning and use of complex data structures or algorithms. It involves non-trivial interactions, such as managing multiple components and optimizing for ef- ficiency. The solution requires significant algorithmic thinking and structured prob- lem decomposition.
Score 5 Description: The problem is ex- tremely complex, requiring sophisticated al- gorithm design, efficient data handling, and advanced techniques. It demands innovative approaches, with intricate component in- teractions and constraints that need careful optimization. Solving it typically requires deep problem-solving skills and adaptabil- ity across programming paradigms."""


from openai import OpenAI
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
client = OpenAI(api_key=GPT_API)
@retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
def create_gpt_completion(qa_, model_engine, client):
    response = client.chat.completions.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": "provide answer"},
            {"role": "user", "content": f"{qa_}"}
        ]
    )
    answer = response.choices[0].message.content
    return answer

def LLM_as_judge(instruction, response, score_rubric):
    score = 0
    prompt = \
f"""
###Task Description:
An instruction (might include an Input in- side it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. 2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
###The instruction to evaluate:
{instruction} ###Response to evaluate:
{response}
###Score Rubric: {score_rubric} 
# ###Feedback:
"""

    score = create_gpt_completion(prompt, MODEL_ENGINE, client)
    
    return score


