from config.config import HOME_DIRECTORY
import json
def provide_gpt4_or_human_example_template(task_name, varient_name, gold_label, question):
    prompt = ''
#------------------------------------------
    if task_name == 'PLAN_BENCH_GENERATION':
        if varient_name == 'gpt4_style_in_context_examples':
            q1 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'
    
            q2 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'

            gold_label1 = \
"""(unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)\n"""

            gold_label2 = \
"""(unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)"""

            a1 = \
"""Initial conditions:\n- Red block is on top of the blue block.\n- Yellow block is on top of the orange block.\n- Both yellow and red blocks are clear.\n- Hand is empty.\n\nGoal:\n- Orange block should be on top of the red block.\n\nPlan steps:\n1. **Unstack the yellow block from the orange block.** \n   - The yellow block is clear and can be unstacked. This action makes the orange block clear. The action is valid.\n   \n2. **Put down the yellow block.** \n   - After unstacking, you're holding the yellow block. You can put it down since the hand is not empty.\n   \n3. **Pick up the orange block.** \n   - The orange block is clear and now on the table. With an empty hand, you can pick it up.\n   \n4. **Stack the orange block on top of the red block.**\n   - The red block is clear after putting down the yellow block. While holding the orange block, you can stack it on the red block.\n\nEach step adheres to the initial conditions and satisfies the goals through logical, permissible operations.

Final Answer: (unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)"""

            a2 = \
"""1. Start by unstacking the blue block from the orange block. This action is possible because the blue block is clear. Now the hand is holding the blue block, and the orange block is clear.\n\n2. Put down the blue block on the table. This clears the orange block completely and empties the hand.\n\n3. Unstack the orange block from the yellow block. This is possible because the orange block is now clear. Now the hand is holding the orange block, and the yellow block is clear.\n\n4. Put down the orange block on the table. This action ensures that the orange block is not disturbing the intended goal and empties the hand. \n\n5. Unstack the yellow block from the red block. This action is possible because the yellow block is clear. The hand now holds the yellow block, and the red block is clear.\n\n6. Stack the yellow block onto the blue block. This ensures that the yellow block is appropriately repositioned, necessary for the subsequent correct arrangement. Now the hand is empty, and the blue block is not clear as the yellow block is on top of it.\n\n7. Pick up the red block. This action is possible because the red block is clear. The hand now holds the red block.\n\n8. Stack the red block on top of the orange block. This places the red block in one part of the final goal, and the hand is now empty.\n\n9. Unstack the yellow block from the blue block. This action is necessary to correctly place the yellow block for the final configuration. The yellow block is now in hand, and the blue block is clear.\n\n10. Stack the yellow block on top of the red block. This places the yellow block in the correct final position, sitting on top of the red block to achieve the goal.

Final Answer: (unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)"""

        if varient_name == 'openai_human_written_examples':
            q1 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'
        
            q2 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'

            gold_label1 = \
"""(unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)\n"""

            gold_label2 = \
"""(unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)\n"""
            a1 = \
"""The goal is to place the orange block on top of the red block. According to the initial conditions, the red block is clear, meaning there is nothing on top of it. Therefore, after we remove the orange block, we can directly place it onto the red block. Based on the initial conditions, the orange block is underneath the yellow block, and the yellow block is clear, meaning there is nothing on top of it. Since our hand is empty, we can directly pick up the yellow block and then put it down on the table. After that, we pick up the orange block and place it on the red block.

Final Answer:
(unstack yellow orange)  
(put-down yellow)  
(pick-up orange)  
(stack orange red)"""
            a2 = \
"""According to the problem, we need to place the red block on top of the orange block. We find that only the red block is on the table. The yellow block is on top of the red block, the orange block is on top of the yellow block, and the blue block is on top of the orange block. The blue block is clear. Therefore, the arrangement order is blue, orange, yellow, and red, with nothing on top of the blue block.

Our goal is to place the red block on top of the orange block and the yellow block on top of the red block. To achieve this, we need to ensure that the bottom block in the final arrangement is the orange block. Based on our initial conditions, the red block is at the bottom, but in our goal, the red block is above the orange block. Therefore, we need to move all the blocks from the table and place them on the table in the correct order, then pick them up one by one and stack them according to the goal's order: yellow > red > orange.

Final Answer:
(unstack blue orange)  
(put down blue)  
(unstack orange yellow)  
(put down orange)  
(unstack yellow red)  
(stack yellow blue)  
(pick up red)  
(stack red orange)  
(unstack yellow blue)  
(stack yellow red)"""

    if task_name == 'PLAN_BENCH_OPTIMALITY':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do:\n\nPick up a block. It takes 1 minute to pick up a block.\nUnstack a block from on top of another block. It takes 1 minute to unstack a block from on top of another block.\nPut down a block. It takes 1 minute to put down a block.\nStack a block on top of another block. It takes 1 minute to stack a block on top of another block.\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\nThe total time to execute the plan is 4 minutes.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN] \nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do:\n\nPick up a block. It takes 1 minute to pick up a block.\nUnstack a block from on top of another block. It takes 1 minute to unstack a block from on top of another block.\nPut down a block. It takes 1 minute to put down a block.\nStack a block on top of another block. It takes 1 minute to stack a block on top of another block.\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\nunstack the orange block from on top of the yellow block\nput down the orange block\nunstack the yellow block from on top of the red block\nstack the yellow block on top of the blue block\npick up the red block\nstack the red block on top of the orange block\nunstack the yellow block from on top of the blue block\nstack the yellow block on top of the red block\n[PLAN END]\nThe total time to execute the plan is 10 minutes.\n\n[STATEMENT]\nAs initial conditions I have that, the yellow block is clear, the hand is empty, the red block is on top of the orange block, the orange block is on top of the blue block, the yellow block is on top of the red block and the blue block is on the table.\nMy goal is to have that the red block is on top of the yellow block and the yellow block is on top of the blue block. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN] \nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = \
"""(unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)"""

        gold_label2 = \
"""(unstack yellow red)\n(put-down yellow)\n(unstack red orange)\n(put-down red)\n(unstack orange blue)\n(put-down orange)\n(pick-up yellow)\n(stack yellow blue)\n(pick-up red)\n(stack red yellow)"""
        
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Reasoning and Inference

#### Initial Setup:
1. **Red block** is on top of the **blue block**, and the red block is **clear**.
2. **Yellow block** is on top of the **orange block**, and the yellow block is **clear**.
3. The **blue block** and **orange block** are on the **table**.
4. The **hand is empty**.

#### Goal:
Move the **orange block** to be on top of the **red block**.

#### Steps to Achieve the Goal:
To achieve the goal, we need to:
1. Unstack the **yellow block** from on top of the **orange block** to free the orange block.
2. Put down the **yellow block**.
3. Pick up the **orange block**.
4. Stack the **orange block** on top of the **red block**.

### Plan and Time Calculation:
1. **Unstack the yellow block from on top of the orange block**:
   - This will take 1 minute. Now, the **orange block** is clear, and the **yellow block** is in hand.

2. **Put down the yellow block**:
   - This will take 1 minute. Now, the **hand is empty**, and the **yellow block** is on the table.

3. **Pick up the orange block**:
   - This will take 1 minute. Now, the **orange block** is in hand.

4. **Stack the orange block on top of the red block**:
   - This will take 1 minute. Now, the **orange block** is on top of the **red block**, and the goal is achieved.

#### Total Time:
The total time required for the plan is **4 minutes**.

---

Final Answer: (unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)
"""

            a2 = \
"""Let's analyze the problem step by step to achieve the goal while minimizing the time. We'll infer the necessary steps based on the rules and constraints.

### Initial Conditions:
1. The **yellow block** is clear.
2. The **hand** is empty.
3. The **red block** is on top of the **orange block**.
4. The **orange block** is on top of the **blue block**.
5. The **yellow block** is on top of the **red block**.
6. The **blue block** is on the table.

### Goal:
- The **red block** is on top of the **yellow block**.
- The **yellow block** is on top of the **blue block**.

### Observations:
- The **yellow block** is currently clear and needs to remain below the **red block** in the final configuration.
- To achieve the goal, we need to:
  1. Rearrange the stack by unstacking the **yellow block** and **red block**.
  2. Place the **yellow block** directly on the **blue block**.
  3. Place the **red block** directly on top of the **yellow block**.

### Plan:
1. **Unstack the yellow block** from on top of the red block (1 minute).
2. **Put down the yellow block** (1 minute).
3. **Unstack the red block** from on top of the orange block (1 minute).
4. **Put down the red block** (1 minute).
5. **Unstack the orange block** from on top of the blue block (1 minute).
6. **Put down the orange block** (1 minute).
7. **Pick up the yellow block** (1 minute).
8. **Stack the yellow block on top of the blue block** (1 minute).
9. **Pick up the red block** (1 minute).
10. **Stack the red block on top of the yellow block** (1 minute).

### Total Time:
10 minutes.

Final Answer: (unstack yellow red)\n(put-down yellow)\n(unstack red orange)\n(put-down red)\n(unstack orange blue)\n(put-down orange)\n(pick-up yellow)\n(stack yellow blue)\n(pick-up red)\n(stack red yellow)
"""

        if varient_name == 'openai_human_written_examples':
            a1 = \
"""We need to use minimum move to achieve our goal. Our goal is to have that the orange block is on top of the red block. According to the initial condition, we know that the red block is clear. In other words, the orange block is not on top of the red block, so we have to move the blocks to achieve the goal. We can directly move the blocks around since our hand is empty. 

Since the the yellow block is on top of the orange block and the yellow block is clear, we can just remove the yellow block from the orange block(unstack it first, then put it down.), then place the orange block on the red block which is clear(pick it up, then stack it). 

Final Answer: (unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)
"""

            a2 = \
"""We need to use minimum move to achieve our goal. One of our goal is to have red block on top f the yellow block, but the initial condition claim that the yellow block is on top of the red block. Therefore, we need to move blocks around to achieve the goal. Since our hand is empty, we can directly move the blocks around. 

According to the initial condition, the blue block is on the table, and the red block is on top of it. However, our goal is having the yellow block on top of the blue block. Therefore, we have to unstack everything on the blue block and stack the yellow and red block back to the blue block.

According to the initial condition, the sequence of the blocks is yellow->red->orange->blue. To clear the blue blcok, we have to upstack and put down the blocks in this sequence: yellow->red->red->orange
Then put them back to the blue block in this sequence: yellow->red.

After taht, the condition will be satisfied.

Final Answer: (unstack yellow red)\n(put-down yellow)\n(unstack red orange)\n(put-down red)\n(unstack orange blue)\n(put-down orange)\n(pick-up yellow)\n(stack yellow blue)\n(pick-up red)\n(stack red yellow)
"""
    
    if task_name == 'PLAN_BENCH_GENERALIZATION':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the yellow block is clear, the white block is clear, the magenta block is clear, the black block is clear, the cyan block is clear, the green block is clear, the violet block is clear, the silver block is clear, the gold block is clear, the hand is empty, the red block is on the table, the blue block is on the table, the orange block is on the table, the yellow block is on the table, the white block is on the table, the magenta block is on the table, the black block is on the table, the cyan block is on the table, the green block is on the table, the violet block is on the table, the silver block is on the table and the gold block is on the table.\nMy goal is to have that the red block is on top of the magenta block, the blue block is on top of the yellow block, the orange block is on top of the green block, the white block is on top of the violet block, the magenta block is on top of the blue block, the black block is on top of the red block, the cyan block is on top of the white block, the green block is on top of the silver block, the violet block is on top of the black block, the silver block is on top of the gold block and the gold block is on top of the cyan block.\n\nMy plan is as follows:\n\n[PLAN]\npick up the blue block\nstack the blue block on top of the yellow block\npick up the magenta block\nstack the magenta block on top of the blue block\npick up the red block\nstack the red block on top of the magenta block\npick up the black block\nstack the black block on top of the red block\npick up the violet block\nstack the violet block on top of the black block\npick up the white block\nstack the white block on top of the violet block\npick up the cyan block\nstack the cyan block on top of the white block\npick up the gold block\nstack the gold block on top of the cyan block\npick up the silver block\nstack the silver block on top of the gold block\npick up the green block\nstack the green block on top of the silver block\npick up the orange block\nstack the orange block on top of the green block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the yellow block is clear, the white block is clear, the black block is clear, the violet block is clear, the hand is empty, the yellow block is on the table, the white block is on the table, the black block is on the table and the violet block is on the table.\nMy goal is to have that the yellow block is on top of the violet block, the white block is on top of the yellow block and the black block is on top of the white block.\n\nMy plan is as follows:\n\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the white block is clear, the black block is clear, the cyan block is clear, the green block is clear, the violet block is clear, the silver block is clear, the gold block is clear, the hand is empty, the red block is on the table, the blue block is on the table, the orange block is on the table, the white block is on the table, the black block is on the table, the cyan block is on the table, the green block is on the table, the violet block is on the table, the silver block is on the table and the gold block is on the table.\nMy goal is to have that the red block is on top of the gold block, the orange block is on top of the silver block, the white block is on top of the cyan block, the black block is on top of the red block, the cyan block is on top of the green block, the green block is on top of the blue block, the violet block is on top of the white block, the silver block is on top of the black block and the gold block is on top of the violet block.\n\nMy plan is as follows:\n\n[PLAN]\npick up the green block\nstack the green block on top of the blue block\npick up the cyan block\nstack the cyan block on top of the green block\npick up the white block\nstack the white block on top of the cyan block\npick up the violet block\nstack the violet block on top of the white block\npick up the gold block\nstack the gold block on top of the violet block\npick up the red block\nstack the red block on top of the gold block\npick up the black block\nstack the black block on top of the red block\npick up the silver block\nstack the silver block on top of the black block\npick up the orange block\nstack the orange block on top of the silver block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the magenta block is clear, the green block is clear, the silver block is clear, the gold block is clear, the hand is empty, the magenta block is on the table, the green block is on the table, the silver block is on the table and the gold block is on the table.\nMy goal is to have that the green block is on top of the silver block, the silver block is on top of the gold block and the gold block is on top of the magenta block.\n\nMy plan is as follows:\n\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = \
"""(pick-up yellow)\n(stack yellow violet)\n(pick-up white)\n(stack white yellow)\n(pick-up black)\n(stack black white)"""

        gold_label2 = \
"""(pick-up gold)\n(stack gold magenta)\n(pick-up silver)\n(stack silver gold)\n(pick-up green)\n(stack green silver)"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

You want to achieve a stacking configuration where the yellow block is on top of the violet block, the white block is on top of the yellow block, and the black block is on top of the white block. Based on your actions and restrictions:

1. **Pick up the yellow block** (it is clear and on the table).
2. **Stack the yellow block on top of the violet block** (the violet block is clear after the previous actions).
3. **Pick up the white block** (it is clear and on the table).
4. **Stack the white block on top of the yellow block** (the yellow block is clear).
5. **Pick up the black block** (it is clear and on the table).
6. **Stack the black block on top of the white block** (the white block is clear).

Final Answer: (pick-up yellow)\n(stack yellow violet)\n(pick-up white)\n(stack white yellow)\n(pick-up black)\n(stack black white)
"""

            a2 = \
"""To solve the problem, let's first analyze the initial conditions and the goal:

- **Initial Conditions**:
  - The magenta, green, silver, and gold blocks are clear and on the table.
  - The hand is empty.

- **Goal**:
  - The green block is on top of the silver block.
  - The silver block is on top of the gold block.
  - The gold block is on top of the magenta block.

### Plan:
1. **Pick up the gold block** (since it is on the table and clear).
2. **Stack the gold block on top of the magenta block**.
3. **Pick up the silver block** (since it is now clear after the gold block is stacked on the magenta block).
4. **Stack the silver block on top of the gold block**.
5. **Pick up the green block** (since it is clear and on the table).
6. **Stack the green block on top of the silver block**.

Final Answer: (pick-up gold)\n(stack gold magenta)\n(pick-up silver)\n(stack silver gold)\n(pick-up green)\n(stack green silver)
"""
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Our hand is empty so we can directly move the blocks around. According to the question, all of the yellow, white, block and violet blocks are clear and all of them are on table, thus we can directly move them around without worry about waht is on top or bottom of them. Our goal is having this sequence: black->white->yellow->violet, thus we can just pick and stack the blocks one by one in this sequence. 
yellow->violet
white->yellow
black->white
Then we will achieve our goal.

Final Answer: (pick-up yellow)\n(stack yellow violet)\n(pick-up white)\n(stack white yellow)\n(pick-up black)\n(stack black white)
"""

            a2 = \
"""Our hand is empty so we can directly move the blocks around. According to the question, all of the magenta, green, silver and gold blocks are clear and all of them are on table, thus we can directly move them around without worry about waht is on top or bottom of them. Our goal is having this sequence: green->silver->gold->magenta, thus we can just pick and stack the blocks one by one in this sequence. 
gold->magenta
silver->gold
green->silver
Then we will achieve our goal.

Final Answer: (pick-up gold)\n(stack gold magenta)\n(pick-up silver)\n(stack silver gold)\n(pick-up green)\n(stack green silver)
"""

    if task_name == 'PLAN_BENCH_REUSE':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n \n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the red block is clear, the orange block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the blue block is on the table, the orange block is on the table and the yellow block is on the table.\nMy plan is as follows:\n\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n \n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the red block is clear, the orange block is clear, the yellow block is clear, the hand is currently holding blue block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy plan is as follows:\n\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = \
"""(unstack yellow orange)\n(put-down yellow)\n"""

        gold_label2 = \
"""(unstack blue orange)\n"""       
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Here’s the reasoning step-by-step to achieve the goal where the red block, orange block, and yellow block are clear, the hand is empty, the red block is on top of the blue block, and the other blocks are on the table:

---

**Initial Conditions Recap:**
1. Red block is clear.
2. Yellow block is clear.
3. Hand is empty.
4. Red block is on top of the blue block.
5. Yellow block is on top of the orange block.
6. Blue block is on the table.
7. Orange block is on the table.

**Goal Conditions Recap:**
1. Red block is clear.
2. Orange block is clear.
3. Yellow block is clear.
4. Hand is empty.
5. Red block is on top of the blue block.
6. Blue block is on the table.
7. Orange block is on the table.
8. Yellow block is on the table.

---

### Reasoning:
1. **Yellow block** is currently on top of the orange block. To make the orange block clear and place the yellow block on the table:
   - Unstack the yellow block from the orange block.
   - Put down the yellow block on the table.

2. After step 1:
   - Orange block becomes clear because no block is on top of it.
   - Yellow block is on the table and clear.

3. **Final Check**:
   - The red block remains clear (as no block is on top of it after step 1).
   - The red block is still on top of the blue block.
   - Blue block remains on the table.
   - Hand is empty after placing the yellow block down.

---

Final Answer: (unstack yellow orange)\n(put-down yellow)\n"""

            a2 = \
"""Let's break down the second [STATEMENT] and the goal step-by-step to infer what needs to happen to achieve the desired state.

### Initial Conditions:
- **Red block** is clear.
- **Blue block** is clear.
- **Yellow block** is clear.
- **Hand** is empty.
- **Blue block** is on top of the orange block.
- **Red block** is on the table.
- **Orange block** is on the table.
- **Yellow block** is on the table.

### Goal:
- **Red block** is clear.
- **Orange block** is clear.
- **Yellow block** is clear.
- **Hand** is holding the **blue block**.
- **Red block** is on the table.
- **Orange block** is on the table.
- **Yellow block** is on the table.

### Plan Inference:
1. **Unstack the blue block from the orange block**:
   - Since the blue block is currently on top of the orange block and the blue block is clear, the first action should be to unstack the blue block from the orange block.
   - After this action:
     - The hand is holding the blue block.
     - The orange block becomes clear.
     - The blue block is no longer on the orange block.

At this point, the goal is already satisfied because:
- The **red block** remains clear.
- The **orange block** becomes clear after the unstacking action.
- The **yellow block** remains clear (it is untouched).
- The **hand** is holding the **blue block**.
- The **red block**, **orange block**, and **yellow block** remain on the table.

Final Answer: (unstack blue orange)\n"""
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""According to the initial condiciton, the condition of red and yellow blocl is clear is already satisfied. The hand is already empty, and the red, blue and orange is already on the table. The red block is already on the blue block. We need to perform some action to make sure the orange block is clear. According to the initial condition, the yellow block is on the orange block. We just need to unstack the yellow block from the orange block so as to make the blue block clear. Since we need to make sure the hand is empty, we can just put-down the yellow block.

Final Answer: (unstack yellow orange)\n(put-down yellow)\n"""

            a2 = \
"""Most of the goal is already satisfied. According to the initial condition, the red and orange blocks are already clear. The red, orange and yellow blocks are already on the table. The only goal that has not been satisfied is make orange clear and use hand to hold the blue block. The orange is not clear because the blue block is on the orange block. so as long as we unstack the blue block, the orange block will be clear and the hand will hold the blue block.

Final Answer: (unstack blue orange)\n"""

    if task_name == 'PLAN_BENCH_VERIFICATION':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the yellow block is on top of the red block, the red block is on the table and the orange block is on the table.\nMy goal is to have that the red block is on top of the blue block and the yellow block is on top of the red block. \nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the red block\nput down the yellow block\npick up the red block\nstack the red block on top of the blue block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is invalid. This is the unmet goal condition:\nthe yellow block is on top of the red block\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the hand is empty, the red block is on top of the blue block, the blue block is on top of the orange block, the orange block is on top of the yellow block and the yellow block is on the table.\nMy goal is to have that the blue block is on top of the orange block, the orange block is on top of the yellow block and the yellow block is on top of the red block. \nMy plan is as follows:\n\n[PLAN]\nunstack the red block from on top of the blue block\nput down the red block\nunstack the blue block from on top of the orange block\nput down the blue block\nunstack the orange block from on top of the yellow block\nstack the orange block on top of the blue block\npick up the yellow block\nstack the yellow block on top of the red block\nunstack the orange block from on top of the blue block\nstack the orange block on top of the yellow block\npick up the blue block\nstack the blue block on top of the orange block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is valid.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the red block is on top of the yellow block and the orange block is on top of the red block. \nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\npick up the red block\nput down the blue block\nstack the orange block on top of the red block\npick up the orange block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is invalid.The following action at step 2 has an unmet precondition:\npick up the red block\nThe unmet precondition is:\nthe hand is empty\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block. \nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nstack the orange block on top of the blue block\npick up the orange block\n[PLAN END]\n\n[VERIFICATION]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the blue block, the yellow block is on top of the orange block, the red block is on the table and the blue block is on the table.\nMy goal is to have that the blue block is on top of the orange block and the yellow block is on top of the red block. \nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nstack the yellow block on top of the red block\nunstack the orange block from on top of the blue block\nput down the orange block\npick up the blue block\nstack the blue block on top of the orange block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is valid.\n\n[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the yellow block is on top of the blue block, the red block is on the table and the orange block is on the table.\nMy goal is to have that the blue block is on top of the red block and the orange block is on top of the blue block. \nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the blue block\nput down the yellow block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is invalid. This is the unmet goal condition:\nthe orange block is on top of the blue block\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the red block is on top of the yellow block, the blue block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the red block is on top of the orange block and the blue block is on top of the yellow block. \nMy plan is as follows:\n\n[PLAN]\nunstack the red block from on top of the yellow block\nstack the blue block on top of the yellow block\npick up the blue block\n[PLAN END]\n\n[VERIFICATION]\nThe above plan is invalid.The following action at step 2 has an unmet precondition:\nstack the blue block on top of the yellow block\nThe unmet precondition is:\nthe hand is currently holding blue block\n[STATEMENT]\nAs initial conditions I have that, the yellow block is clear, the hand is empty, the red block is on top of the orange block, the orange block is on top of the blue block, the yellow block is on top of the red block and the blue block is on the table.\nMy goal is to have that the red block is on top of the blue block and the yellow block is on top of the orange block. \nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the red block\nput down the yellow block\nunstack the red block from on top of the orange block\nstack the red block on top of the yellow block\nunstack the orange block from on top of the blue block\nput down the orange block\nunstack the red block from on top of the yellow block\nstack the red block on top of the blue block\npick up the yellow block\nstack the yellow block on top of the orange block\n[PLAN END]\n\n[VERIFICATION]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = \
"""The above plan is invalid. The following action at step 2 has unmet preconditions:\nstack the orange block on top of the blue block\nThe unmet preconditions are:\nthe blue block is clear and the hand is currently holding orange block"""

        gold_label2 = \
"""The above plan is valid.\n"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""To verify and finalize the plan, let's evaluate step by step based on the provided rules and conditions.

---

### Initial Conditions:
- **Blocks**: The red block is clear, the blue block is clear, the yellow block is clear.
- **Hand**: Empty.
- **Stacking**: The blue block is on top of the orange block.
- **Table**: The red block, orange block, and yellow block are on the table.
- **Goal**: The orange block is on top of the blue block.

---

### Plan Verification:

#### Step 1: **Unstack the blue block from on top of the orange block.**
- **Preconditions**: The blue block is clear, and the hand is empty.
- **Result**: The hand now holds the blue block, and the orange block becomes clear.

#### Step 2: **Stack the orange block on top of the blue block.**
- **Preconditions**: The orange block is clear, and the hand is holding the blue block.
- **Violation**: This step fails because the hand is not holding the orange block; it is holding the blue block. You cannot stack the orange block at this moment.

#### Step 3: **Pick up the orange block.**
- **Preconditions**: This action is not valid because the hand is already holding the blue block, violating the condition that the hand must be empty to pick up a block.

---

### Issues in the Plan:
The plan is invalid because:
1. In step 2, the blue block cannot be stacked as the plan misinterprets which block is being held.
2. In step 3, the orange block cannot be picked up as the hand is not empty.

Final Answer: The above plan is invalid. The following action at step 2 has unmet preconditions:\nstack the orange block on top of the blue block\nThe unmet preconditions are:\nthe blue block is clear and the hand is currently holding orange block"""

            a2 = \
"""### Inference and Verification

#### Initial Conditions:
- Yellow block is clear.
- Hand is empty.
- Red block is on top of the orange block.
- Orange block is on top of the blue block.
- Yellow block is on top of the red block.
- Blue block is on the table.

#### Goal:
- Red block is on top of the blue block.
- Yellow block is on top of the orange block.

#### Plan Analysis:
1. **Unstack the yellow block from on top of the red block.**
   - Valid: Yellow block is clear, and the hand is empty.
   - After action: Hand holds the yellow block, and the red block becomes clear.

2. **Put down the yellow block.**
   - Valid: Hand is holding the yellow block.
   - After action: Yellow block is on the table, and the hand becomes empty.

3. **Unstack the red block from on top of the orange block.**
   - Valid: Red block is clear, and the hand is empty.
   - After action: Hand holds the red block, and the orange block becomes clear.

4. **Stack the red block on top of the yellow block.**
   - Valid: Hand is holding the red block, and the yellow block is clear.
   - After action: Red block is on top of the yellow block, and the hand becomes empty.

5. **Unstack the orange block from on top of the blue block.**
   - Valid: Orange block is clear, and the hand is empty.
   - After action: Hand holds the orange block, and the blue block becomes clear.

6. **Put down the orange block.**
   - Valid: Hand is holding the orange block.
   - After action: Orange block is on the table, and the hand becomes empty.

7. **Unstack the red block from on top of the yellow block.**
   - Valid: Red block is clear, and the hand is empty.
   - After action: Hand holds the red block, and the yellow block becomes clear.

8. **Stack the red block on top of the blue block.**
   - Valid: Hand is holding the red block, and the blue block is clear.
   - After action: Red block is on top of the blue block, and the hand becomes empty.

9. **Pick up the yellow block.**
   - Valid: Yellow block is clear, and the hand is empty.
   - After action: Hand holds the yellow block.

10. **Stack the yellow block on top of the orange block.**
    - Valid: Hand is holding the yellow block, and the orange block is clear.
    - After action: Yellow block is on top of the orange block, and the hand becomes empty.

#### Verification:
- At the end of the plan:
  - Red block is on top of the blue block.
  - Yellow block is on top of the orange block.
- All goal conditions are satisfied.

---

Final Answer: The above plan is valid.\n"""
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""The goal is to have the orange block on the blue block. According to the initial condition, the blue block is on the orange block. To do so, we need to unstack the orange block, put it on the table then stack the blue block on the orange block. However, the given plan directly put the orange block on the blue block on the second step, with is not valid since the blue block is hold in hand after the first step. 

Final Answer: The above plan is invalid. The following action at step 2 has unmet preconditions:\nstack the orange block on top of the blue block\nThe unmet preconditions are:\nthe blue block is clear and the hand is currently holding orange block"""

            a2 = \
"""1. **Unstack the yellow block from on top of the red block.**
    - Valid: 

2. **Put down the yellow block.**
    - Valid: 

3. **Unstack the red block from on top of the orange block.**
    - Valid: 

4. **Stack the red block on top of the yellow block.**
    - Valid: 

5. **Unstack the orange block from on top of the blue block.**
    - Valid: 

6. **Put down the orange block.**
    - Valid: 

7. **Unstack the red block from on top of the yellow block.**
    - Valid: 

8. **Stack the red block on top of the blue block.**
    - Valid: 

9. **Pick up the yellow block.**
    - Valid: 

10. **Stack the yellow block on top of the orange block.**
    - Valid: 

Every steps is valid.

Final Answer: The above plan is valid.\n"""

    if task_name == 'PLAN_BENCH_REPLANING':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\nDuring execution, an unexpected event has occurred.\nAfter executing the action \"pick up the orange block\" at step 3 in the plan, the following facts unexpectedly became true: the orange block is clear, the hand is empty and the orange block is on top of the red block\nThe following facts became unexpectedly false: the red block is clear and the hand is currently holding orange block\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]\nunstack the orange block from on top of the red block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n\nDuring execution, an unexpected event has occurred.\nAfter executing the action \"pick up the orange block\" at step 3 in the plan, the following facts unexpectedly became true: the orange block is clear, the hand is empty and the orange block is on top of the red block\nThe following facts became unexpectedly false: the red block is clear and the hand is currently holding orange block\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the hand is empty, the red block is on top of the yellow block, the blue block is on top of the orange block, the yellow block is on top of the blue block and the orange block is on the table.\nMy goal is to have that the red block is on top of the orange block.\nMy plan is as follows:\n\n[PLAN]\nunstack the red block from on top of the yellow block\nput down the red block\nunstack the yellow block from on top of the blue block\nput down the yellow block\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the red block\nstack the red block on top of the orange block\n[PLAN END]\n\nDuring execution, an unexpected event has occurred.\nAfter executing the action \"pick up the red block\" at step 7 in the plan, the following facts unexpectedly became true: the red block is clear, the hand is empty and the red block is on top of the blue block\nThe following facts became unexpectedly false: the blue block is clear and the hand is currently holding red block\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]\nunstack the red block from on top of the blue block\nstack the red block on top of the orange block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the red block is on top of the blue block, the blue block is on top of the yellow block and the yellow block is on top of the orange block.\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nstack the blue block on top of the red block\npick up the yellow block\nstack the yellow block on top of the orange block\nunstack the blue block from on top of the red block\nstack the blue block on top of the yellow block\npick up the red block\nstack the red block on top of the blue block\n[PLAN END]\n\nDuring execution, an unexpected event has occurred.\nAfter executing the action \"unstack the blue block from on top of the red block\" at step 5 in the plan, the following facts unexpectedly became true: the blue block is clear, the hand is empty and the blue block is on top of the yellow block\nThe following facts became unexpectedly false: the yellow block is clear and the hand is currently holding blue block\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = ''

        gold_label2 = \
"""(pick-up red)\n(stack red blue)\n"""
        
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Let's walk through what happened and what the correct response should be.

---

### **Initial Conditions (Second Statement):**

* The red block is **clear**.
* The yellow block is **clear**.
* The **hand is empty**.
* The **red block is on top of the blue block**.
* The **yellow block is on top of the orange block**.
* The **blue block** and **orange block** are **on the table**.

### **Goal:**

* The **orange block is on top of the red block**.

### **Original Plan:**

1. Unstack the yellow block from on top of the orange block
2. Put down the yellow block
3. Pick up the orange block
4. Stack the orange block on top of the red block

### **Unexpected Event After Step 3 ("pick up the orange block"):**

* Unexpectedly became **true**:

  * The orange block is **clear**
  * The hand is **empty**
  * The orange block is **on top of the red block**
* Unexpectedly became **false**:

  * The red block is **clear**
  * The hand is currently **holding orange block**

---

### **Inference:**

These unexpected facts suggest that **instead of holding** the orange block after picking it up, **it was somehow automatically placed on top of the red block**, possibly due to an external or environmental action. This means that the **goal state** (“orange block is on top of red block”) has already been **achieved**, even though it happened in an unexpected way.

We now analyze the current world state:

* **orange block is on top of red block** ✅
* **red block is no longer clear** ✅ (makes sense, because something is now on top)
* **hand is empty** ✅
* **orange block is clear** ✅ (nothing on top)

### **Conclusion:**

Since the **goal condition is already satisfied**, **no further actions are needed**.

---

Final Answer: ()"""

            a2 = \
"""Let's walk through what happened and reason out the correct replan step by step.

---

### Current Known State After the Unexpected Event:

* `blue block is clear` ✅
* `yellow block is *not* clear` ❌
* `blue block is on top of yellow block` ✅
* `hand is empty` ✅
* `red block is on the table` (from earlier context) ✅
* `yellow block is on the table` ✅
* `orange block is on the table` ✅

**Goal**:

1. red block on top of blue block
2. blue block on top of yellow block
3. yellow block on top of orange block

---

### Current Block Configuration (Stacked from bottom to top):

* orange
* yellow (on top of orange)
* blue (on top of yellow)
* red (on the table)

---

### What needs to happen:

We already *accidentally* achieved:

* ✅ blue on yellow
* ✅ yellow on orange

We need to **place red on top of blue**.

To do that:

* `red` is on the table and **clear**
* `blue` is currently **clear**, sitting on yellow
  → So we can directly pick up red and stack it on blue.

---

### Final Answer: (pick-up red)\n(stack red blue)\n"""
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""
The unexpect facts suggest that instead of holding the orange block after pick it up, it somehow being placed on top of the red block. I believe what possibly happen might be the orange block was droped on the red block when holding it. after the unexpected facts happens, the current world state becomes 

orange block is on top of red block
red block is no longer clear
hand is empty
orange block is clear

this state meets the goal condition: the orange block is on top of the red block.

Final Answer: ()"""

            a2 = \
"""our goal is having
1. red block on top of blue block
2. blue block on top of yellow block
3. yellow block on top of orange block

our current states are 


### Current Block Configuration (Stacked from bottom to top):

1. orange on the table
2. yellow on top of orange
3. red on the table
4. blue on top of yellow
5. orange, red and blue are clear since they are on the top of other blocks or table.

in the new states, we already have achieved

1. blue on yellow
2. yellow on orange

we need a new plan to achieve red on top of blue.

to do that, we can just pick up the red block since it is clear, then place it on top of the blue block which is also clear

Final Answer: (pick-up red)\n(stack red blue)\n"""
    
    if task_name == 'PLAN_BENCH_EXECUTION':
        q1 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block and the red block is on the table.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]\nunstack the orange block from on top of the blue block\nput down the orange block\nunstack the blue block from on top of the red block\nput down the blue block\npick up the red block\n[ACTION SEQUENCE END]\n[RESULTING STATE]\nthe blue block is clear, the orange block is clear, the hand is currently holding red block, the blue block is on the table and the orange block is on the table\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the red block is on the table, the blue block is on the table and the orange block is on the table.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]\npick up the red block\n[ACTION SEQUENCE END]\n[RESULTING STATE]\n\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""
        
        q2 = \
"""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the red block is on the table, the blue block is on the table and the orange block is on the table.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]\npick up the red block\nstack the red block on top of the orange block\n[ACTION SEQUENCE END]\n[RESULTING STATE]\nthe red block is clear, the blue block is clear, the hand is empty, the red block is on top of the orange block, the blue block is on the table and the orange block is on the table\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the red block and the red block is on the table.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]\nunstack the blue block from on top of the orange block\nput down the blue block\nunstack the orange block from on top of the red block\nput down the orange block\n[ACTION SEQUENCE END]\n[RESULTING STATE]\n\nPlease inference first then provide the final plan at the end after the word 'Final Answer:'"""

        gold_label1 = \
"""the blue block is clear, the orange block is clear, the hand is currently holding red block, the blue block is on the table, the orange block is on the table"""

        gold_label2 = \
"""the red block is on the table, the blue block is clear, the orange block is clear, the red block is clear, the blue block is on the table, the orange block is on the table, the hand is empty"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

We begin with the following initial conditions:

* red block is clear, on the table
* blue block is clear, on the table
* orange block is clear, on the table
* hand is empty

---

**Action 1**: `pick up the red block`

* **Preconditions check**:

  * Hand is empty ✅
  * Red block is on the table ✅
  * Red block is clear ✅
    → Action is **valid**

* **Effects**:

  * Hand is now holding the red block
  * Red block is no longer on the table
  * Hand is no longer empty

---

### \[RESULTING STATE]

After executing the action `pick up the red block`, the new state is:

* **Hand is holding red block**
* **Blue block is clear**
* **Orange block is clear**
* **Blue block is on the table**
* **Orange block is on the table**
* **Red block is not on the table**
* **Red block is being held**

---

✅ All state transitions are consistent with the action rules and constraints.

---

**Final Answer:**

the hand is currently holding red block, the blue block is clear, the orange block is clear, the blue block is on the table, and the orange block is on the table"""

            a2 = \
"""Let's analyze the scenario step-by-step based on the **initial conditions**, **action sequence**, and the **rules** of block manipulation.

---

### 🔹 Initial Conditions:

* blue block is **clear**
* hand is **empty**
* blue is on **orange**
* orange is on **red**
* red is on the **table**

This implies a **stack**:

```
top:    blue
        orange
bottom: red (on table)
```

---

### 🔹 Action Sequence:

1. **unstack the blue block from on top of the orange block**

   * Preconditions:

     * blue must be on orange ✅
     * blue must be clear ✅
     * hand must be empty ✅
   * Result:

     * hand is now **holding blue**
     * orange is now **clear**
     * blue is **not on orange** anymore

2. **put down the blue block**

   * Preconditions:

     * hand must be holding a block ✅
   * Result:

     * blue is now **on table**
     * hand is now **empty**
     * blue is **clear**

3. **unstack the orange block from on top of the red block**

   * Preconditions:

     * orange must be on red ✅
     * orange must be clear ✅ (it became clear after blue was removed)
     * hand must be empty ✅
   * Result:

     * hand is now **holding orange**
     * red is now **clear**
     * orange is no longer on red

4. **put down the orange block**

   * Preconditions:

     * hand must be holding a block ✅
   * Result:

     * orange is now on table
     * hand is now empty
     * orange is clear

---

### 🔹 Final Resulting State:

After executing all actions:

* **red block** is on table and is **clear**
* **orange block** is on table and is **clear**
* **blue block** is on table and is **clear**
* **hand** is **empty**

All three blocks are on the table and clear, and the hand is empty.

---

**Final Answer:**
the red block is clear, the orange block is clear, the blue block is clear, the hand is empty, the red block is on the table, the orange block is on the table, and the blue block is on the table."""

        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Initially, the red, blue and orange blocks are all on the table and they are clear. The hand is also clear. After pick up the red block, the hand is holding the red block so it is not on the table anymore. The states of the other blocks are not affected. 

Final Answer: the blue block is clear, the orange block is clear, the hand is currently holding red block, the blue block is on the table, the orange block is on the table
"""

            a2 = \
"""Initially, blue block is clear and it is on top of the orange block. orange block is on top of the red block. red block is on the table. there is nothing on the blue block. there is no other block. after excute the action in sequence, the blue, orange and red blocks are placed on the table in seuence one by one. eventually, each individual blocks are on the table and no blocks is on another blocks, thus they are all clear. the hand is not holding any blocks. 

Final Answer: the red block is on the table, the blue block is clear, the orange block is clear, the red block is clear, the blue block is on the table, the orange block is on the table, the hand is empty"""

    if task_name == 'BOOLQ':
        q1 = \
"""Given the context: Persian (/\u02c8p\u025c\u02d0r\u0292\u0259n, -\u0283\u0259n/), also known by its endonym Farsi (\u0641\u0627\u0631\u0633\u06cc f\u0101rsi (f\u0252\u02d0\u027e\u02c8si\u02d0) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the context: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.\n\nis elder scrolls online the same as skyrim?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"""
        gold_label1 = \
"""True"""
        gold_label2 = \
"""False"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Persian, also known as Farsi in Iran and Dari in Afghanistan, is the same language with regional differences in dialect and vocabulary. Both Iran and Afghanistan officially recognize this language under different names, and it is primarily the same language family with slight regional variations. This supports the idea that they speak the same language.

Final Answer: True"""
            a2 = \
"""The Elder Scrolls Online (ESO) and The Elder Scrolls V: Skyrim are both set on the same continent, Tamriel, and share similarities in world structure and themes, such as having concurrent conflicts involving power struggles and the fate of the world. However, ESO is set approximately 1,000 years before Skyrim and is structured as an MMORPG (Massively Multiplayer Online Role-Playing Game), whereas Skyrim is a single-player RPG. The main conflicts and gameplay focus in ESO revolve around the Daedric Prince Molag Bal and the battle for the imperial throne, distinct from the primary narrative of Skyrim, which centers around the Dragonborn's battle against Alduin and personal exploration of Skyrim. Thus, while they share the same universe and certain thematic elements, they differ significantly in timeline, gameplay style, and narrative focus.

Final Answer: False"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""The context mentions that "It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran." The 'it' in this sentence represents Persian. This sentene mention that Persian is spoken in Iran and Afghnistan. Therefore, the answer is True.

Final Answer: True"""
            a2 = \
"""The context mentions "It has a broadly similar structure to Skyrim". In this sentence, it represents the Elder Scrolls series. This sentence tell us that The Elder Scrolls series is different game from Skyrim, since the sentence compare it to Skyrim. Therefore, the answer is False.

Final Answer: False"""
    
    if task_name == 'HELLASWAG':
        q1 = \
"""Please complete the paragraph.\n\"Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then\"\n Here is the options\n(1): \", the man adds wax to the windshield and cuts it.\"\n(2): \", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.\"\n(3): \", the man puts on a christmas coat, knitted with netting.\"\n(4): \", the man continues removing the snow on his car.\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (1/2/3/4) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Please complete the paragraph.\n\"A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans\"\n Here is the options\n(1): \"contain egg yolks and baking soda.\"\n(2): \"are then sprinkled with brown sugar.\"\n(3): \"are placed in a strainer on the counter.\"\n(4): \"are filled with pastries and loaded into the oven.\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (1/2/3/4) at the end, after 'Final Answer:'\n"""
        gold_label1 = \
"""4"""
        gold_label2 = \
"""4"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

1. **Option (1):** "The man adds wax to the windshield and cuts it."  
   - This does not align well with the initial context. Writing over the snow on the window does not suggest that the man is preparing the windshield for waxing or cutting.

2. **Option (2):** "A person boards a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled."  
   - This option introduces multiple unrelated actions and characters, making it incoherent and disconnected from the original context of a man writing on a snow-covered window and a smiling woman.

3. **Option (3):** "The man puts on a Christmas coat, knitted with netting."  
   - This option is also inconsistent with the context. The focus is on writing on the snow and the interaction with the woman, not about the man's clothing.

4. **Option (4):** "The man continues removing the snow on his car."  
   - This aligns best with the context. Writing on snow-covered windows could reasonably be part of clearing the snow from the car, and the progression of actions matches the narrative flow.

Final Answer: 4"""

            a2 = \
"""### Inference:

1. **Option (1):** "The pans contain egg yolks and baking soda."  
   - This could potentially make sense in a baking context. However, the paragraph mentions that the chef is "presenting" the pans, and there is no mention of specific ingredients like egg yolks or baking soda.

2. **Option (2):** "The pans are then sprinkled with brown sugar."  
   - While sprinkling brown sugar could occur in a baking process, it doesn't logically follow from the act of showing and presenting the pans. This action feels premature given the context.

3. **Option (3):** "The pans are placed in a strainer on the counter."  
   - This does not align well with the context. Presenting a stack of baking pans in a large kitchen seems unrelated to placing them in a strainer, which is not a common or logical step in this scenario.

4. **Option (4):** "The pans are filled with pastries and loaded into the oven."  
   - This makes the most sense. After showing and presenting the baking pans, the logical next step in a kitchen context is to fill them with pastries (suggesting preparation for baking) and place them in the oven.

Final Answer: 4"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""The option 1 does not make sense. The scene is very wired. The scene in option 2 is unrelated to the scene described in the question. The scene in option 3 is unrelated to the scene described in the question. The scene in the option 4 seems consistent with the scene described in the question.

Final Answer: 4"""
            a2 = \
"""The baking pan in a kitchen usually do not contain egg yorks and baking soda. Those kind of food are usally in containers, not baking pan. Thus option 1 is wrong. The option 2 describe a scene of 'sprinkled with brown sugar'. This scene is not logically consistent with the scene described in the question. Thus option 2 is wrong. The option 3 describes where the pan is, but the question says the chief present the pan. If the chief show the pan, it is likely he wants to show the pan or the food in the pan, but not place the pan. The option 4 describe what is inside the pan, then the pan is loaded into the ven. this is logically consistent with the scene in the question.

Final Answer: 4"""

    if task_name == 'ARC_CHALLENGE':
        q1 = \
"""We have the question.\n\"George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\"\n Here is the options\n(A): \"dry palms\"\n(B): \"wet palms\"\n(C): \"palms covered with oil\"\n(D): \"palms covered with lotion\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""We have the question.\n\"Which of the following statements best explains why magnets usually stick to a refrigerator door?\"\n Here is the options\n(A): \"The refrigerator door is smooth.\"\n(B): \"The refrigerator door contains iron.\"\n(C): \"The refrigerator door is a good conductor.\"\n(D): \"The refrigerator door has electric wires in it.\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""A"""
        gold_label2 = \
"""B"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:
When rubbing hands to generate heat, the primary factor is friction. The more friction there is between the surfaces, the more heat will be produced. Let's analyze each option:

1. **Dry palms (A):** Dry surfaces generally have higher friction compared to wet or lubricated surfaces. Therefore, they will produce the most heat when rubbed together.

2. **Wet palms (B):** Water reduces friction by acting as a lubricant, so less heat will be generated compared to dry palms.

3. **Palms covered with oil (C):** Oil is a lubricant, and it significantly reduces friction, further decreasing the heat generated compared to dry or even wet palms.

4. **Palms covered with lotion (D):** Lotion is also a lubricant, similar to oil, and reduces friction, resulting in less heat generation.

Final Answer: A"""

            a2 = \
"""### Inference:
Magnets stick to certain materials because of their magnetic properties. Magnets are attracted to materials that can be magnetized, such as those containing iron, nickel, or cobalt. Let’s evaluate the options:

1. **(A): "The refrigerator door is smooth."**  
   Surface smoothness has no effect on whether a magnet sticks. This is irrelevant to the magnetic attraction.

2. **(B): "The refrigerator door contains iron."**  
   This is correct. The refrigerator door typically contains steel, which is an alloy of iron. Iron is a ferromagnetic material, and magnets are strongly attracted to it.

3. **(C): "The refrigerator door is a good conductor."**  
   While metals are generally good conductors, conductivity is not the reason magnets stick. Magnets do not rely on electrical properties for attraction.

4. **(D): "The refrigerator door has electric wires in it."**  
   This is incorrect. The presence of wires does not explain magnetic attraction.

Final Answer: B"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""The more friction is on the hand, the more heat will be generated. The dry palms have larger friction than the other options. Thus we should choose A.

Final Answer: A"""
            a2 = \
"""The magnets usally stick to the refrigerator door because of magnetic force. Either the door is smooth, the door is a good conductor or the door has electric wires do not related to magnetic force. If the door contains iron, then it has the property to attract magnets. Thus the option is B

Final Answer: B"""

    if task_name == 'MMLU_PRO':
        q1 = \
"""We have the question.\n\"What is the significance of the 1:2:4 resonance in the Jupiter's moons system?\"\n Here is the options\n(A): \"It causes the three moons to always be in a straight line.\"\n(B): \"It causes Io to rotate on its axis faster than it orbits Jupiter\"\n(C): \"It creates a gap with no asteriods between the orbits.\"\n(D): \"It results in the formation of a visible ring around Jupiter.\"\n(E): \"It prevents the moons from colliding with each other.\"\n(F): \"It causes a gravitational pull that affects Jupiter's orbit around the sun.\"\n(G): \"It results in a uniform gravitational pull on all three moons.\"\n(H): \"It prevents formation of the ring material into other moons.\"\n(I): \"It makes the orbit of Io slightly elliptical.\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""We have the question.\n\"We were first able to accurately measure the diameter of Pluto from:\"\n Here is the options\n(A): \"Lunar-based observations made during NASA's Apollo missions\"\n(B): \"The Voyager 2 spacecraft flyby in the 1980s\"\n(C): \"Observations made by the Chandra X-ray Observatory\"\n(D): \"brightness measurements made during mutual eclipses of Pluto and Charon\"\n(E): \"The Mars Rover's telescope observations of Pluto\"\n(F): \"radar observations made by the Arecibo telescope\"\n(G): \"Observations made by the Spitzer Space Telescope\"\n(H): \"Hubble Space Telescope images that resolved Pluto's disk\"\n(I): \"a New Horizons flyby in the 1990s\"\n(J): \"Images captured by the Kepler Space Telescope\"\n\nWhat is the correct option?\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'\n"""
        gold_label1 = \
"""I"""
        gold_label2 = \
"""D"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:
The 1:2:4 resonance in the Jupiter moon system refers to the orbital resonance among the three Galilean moons: Io, Europa, and Ganymede. This resonance means:

- **For every one orbit Ganymede completes**, Europa completes **two orbits**, and Io completes **four orbits**.
- This resonance is significant because of the gravitational interactions between these moons, which have important consequences.

Let's evaluate the options:

1. **(A): "It causes the three moons to always be in a straight line."**  
   This is incorrect. The resonance does not cause the moons to align in a straight line, though their orbits are related.

2. **(B): "It causes Io to rotate on its axis faster than it orbits Jupiter."**  
   Incorrect. Io's rotation is tidally locked to its orbit, meaning its rotation matches its orbital period around Jupiter, which is unrelated to the 1:2:4 resonance.

3. **(C): "It creates a gap with no asteroids between the orbits."**  
   Incorrect. This statement does not apply to the moons of Jupiter but is more relevant to orbital resonances in the asteroid belt (e.g., Kirkwood gaps).

4. **(D): "It results in the formation of a visible ring around Jupiter."**  
   Incorrect. Rings around Jupiter are primarily formed from dust and particles ejected from moons, not due to orbital resonances.

5. **(E): "It prevents the moons from colliding with each other."**  
   Partially true, but not the primary significance. The resonance stabilizes their orbits, but direct collision prevention is not its defining role.

6. **(F): "It causes a gravitational pull that affects Jupiter's orbit around the sun."**  
   Incorrect. The gravitational influence of these moons on Jupiter’s orbit around the Sun is negligible.

7. **(G): "It results in a uniform gravitational pull on all three moons."**  
   Incorrect. The resonance causes periodic variations in gravitational forces, not uniformity.

8. **(H): "It prevents formation of the ring material into other moons."**  
   Incorrect. This resonance concerns existing moons, not ring material.

9. **(I): "It makes the orbit of Io slightly elliptical."**  
   Correct. The resonance causes periodic gravitational interactions, particularly between Io, Europa, and Ganymede. These interactions maintain Io's slightly elliptical orbit. This is important because it leads to tidal heating within Io, driving its volcanic activity.

Final Answer: I
"""
            a2 = \
"""To determine the correct answer, we analyze the question and options:

### Analysis:
- **(A): Lunar-based observations made during NASA's Apollo missions**  
  The Apollo missions focused on lunar exploration, not observations of distant objects like Pluto.

- **(B): The Voyager 2 spacecraft flyby in the 1980s**  
  Voyager 2 did not visit Pluto; it explored the outer planets (Jupiter, Saturn, Uranus, Neptune).

- **(C): Observations made by the Chandra X-ray Observatory**  
  Chandra focuses on X-ray observations, unsuitable for accurately measuring Pluto’s diameter.

- **(D): Brightness measurements made during mutual eclipses of Pluto and Charon**  
  This method was instrumental in accurately determining Pluto's size when Charon was discovered in 1978.

- **(E): The Mars Rover's telescope observations of Pluto**  
  The Mars Rovers are designed for Mars exploration, not observations of distant celestial bodies like Pluto.

- **(F): Radar observations made by the Arecibo telescope**  
  Radar observations require a significant reflection signal, which Pluto's distant and icy surface does not provide effectively.

- **(G): Observations made by the Spitzer Space Telescope**  
  Spitzer specializes in infrared observations, useful for studying thermal properties, not diameter measurements.

- **(H): Hubble Space Telescope images that resolved Pluto's disk**  
  Hubble captured high-resolution images of Pluto, which helped resolve its disk and study surface details, but this was after mutual eclipse observations.

- **(I): A New Horizons flyby in the 1990s**  
  The New Horizons mission flew by Pluto in 2015, not the 1990s, so this option is incorrect.

- **(J): Images captured by the Kepler Space Telescope**  
  Kepler is designed to detect exoplanets, not measure Pluto's diameter.

### Inference:
The first accurate measurement of Pluto’s diameter was achieved using **brightness measurements during the mutual eclipses of Pluto and Charon** (option D). These eclipses provided valuable information about the sizes and orbital characteristics of both Pluto and Charon.

Final Answer: D"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""In the Jupiter moon system, the 1:2:4 resonance refers to the orbital resonance relationship between Io, Europa, and Ganymede. Specifically, for every orbit Ganymede completes around Jupiter, Europa completes two orbits, and Io completes four.

**Analysis of Incorrect Options**:  
   - (A) Incorrect: The 1:2:4 resonance does not mean the moons are always aligned in a straight line.  
   - (B) Incorrect: Io’s rotation is tidally locked with its orbit, meaning its rotation and orbital periods are the same.  
   - (C) Incorrect: Orbital resonance does not directly create gaps between orbits devoid of asteroids.  
   - (D) Incorrect: Jupiter's ring system is primarily composed of small particles and is not related to the moon resonance.  
   - (F) Incorrect: The resonance does not significantly impact Jupiter’s orbit around the Sun.  
   - (G) Incorrect: The gravitational pull is not uniformly distributed among the three moons.  
   - (H) Incorrect: The formation of ring material into other moons is influenced by other factors, unrelated to the 1:2:4 resonance.

The main effects of the 1:2:4 resonance are preventing the moons from colliding with each other and causing Io’s orbit to become slightly elliptical. Thus both E and I are plausible. However, the primary significance of the 1:2:4 resonance is maintaining orbital eccentricity and the associated effects, favoring option (I).

Final Answer: I"""
            a2 = \
"""To deduce the correct answer, we need to analyze the scientific feasibility and historical context of each option:

1. **(A): "Lunar-based observations made during NASA's Apollo missions"**  
   - The Apollo missions focused on the Moon and did not involve detailed observations of distant celestial bodies like Pluto. Therefore, this can be eliminated.

2. **(B): "The Voyager 2 spacecraft flyby in the 1980s"**  
   - Voyager 2 flew by outer planets (like Jupiter, Saturn, Uranus, and Neptune) but did not visit Pluto. This option can also be eliminated.

3. **(C): "Observations made by the Chandra X-ray Observatory"**  
   - The Chandra X-ray Observatory is primarily used for studying high-energy phenomena, such as black holes and supernova remnants. Pluto's low-temperature environment would not emit enough X-rays for measurement, so this option is not viable.

4. **(D): "Brightness measurements made during mutual eclipses of Pluto and Charon"**  
   - During mutual eclipses between Pluto and its moon Charon, variations in brightness can indeed be used to calculate diameters. This is a plausible option.

5. **(E): "The Mars Rover's telescope observations of Pluto"**  
   - Mars rovers are designed to study the Martian surface and environment, not to observe distant objects like Pluto. This option is incorrect.

6. **(F): "Radar observations made by the Arecibo telescope"**  
   - The Arecibo telescope is primarily used for observing near-Earth objects. Given Pluto's extreme distance, radar signals would be insufficient to measure its diameter. This option can be ruled out.

7. **(G): "Observations made by the Spitzer Space Telescope"**  
   - While the Spitzer Space Telescope observes in the infrared spectrum and studies thermal emissions, it does not have the resolution to accurately measure Pluto's diameter. This option is unlikely.

8. **(H): "Hubble Space Telescope images that resolved Pluto's disk"**  
   - The Hubble Space Telescope was the first to resolve Pluto's disk in detail, allowing for the accurate measurement of its diameter. This makes it a strong candidate.

9. **(I): "A New Horizons flyby in the 1990s"**  
   - The New Horizons spacecraft flew by Pluto in 2015, not the 1990s. Thus, this option is incorrect.

10. **(J): "Images captured by the Kepler Space Telescope"**  
    - The Kepler Space Telescope is designed for detecting exoplanets by observing light curves of stars, not for measuring Pluto. This option is also incorrect.

### Conclusion:  
The most plausible answers are **(D)** and **(H)**. However, the first accurate measurement of Pluto's diameter is widely attributed to images taken by the Hubble Space Telescope.

Final Answer: H"""

    if task_name == 'MMLU_PRO_LAW':
        q1 = \
"""A federal grand jury was investigating a corporation whose tanker ship had spilled crude oil into environmentally sensitive waters. The grand jury issued a subpoena requiring the corporation to produce all emails and internal documents regarding the corporation's knowledge of the risks of an oil spill. The corporation has objected, citing its Fifth Amendment privilege against self-incrimination. Can the subpoena be enforced?\nAvailable options are the following:\nA: Yes, because the Fifth Amendment privilege only applies to personal testimonies.\nB: Yes, because a corporation has no Fifth Amendment privilege.\nC: No, because the corporation was not granted use-and-derivative-use immunity.\nD: No, because the corporation was not granted transactional immunity.\nE: Yes, because the Fifth Amendment privilege does not apply to the compelled production of documents.\nF: No, because the documents are protected under attorney-client privilege.\nG: No, because a corporation has the same Fifth Amendment rights as an individual.\nH: No, because the subpoena violates the corporation's Fourth Amendment rights.\nI: Yes, because the documents are not self-incriminating.\nJ: Yes, because the subpoena is part of a lawful investigation.\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""The defendant was walking down the street when he saw a woman struggling with a man over a briefcase. Unbeknownst to the defendant, the woman had just stolen the briefcase from the man on the street. Believing the woman to be the victim of an attack, the defendant intervened and punched the man until the woman was able to get away with the briefcase. Confused as to why he wasn't being hailed as a hero by the woman, the defendant eventually realized that he had been an unwitting accomplice to the woman's theft. The defendant apologized profusely to the man and went home. According to the alter ego rule, which of the following statements is correct with respect to the amount of force that the defendant was entitled to use in the woman's defense?\nAvailable options are the following:\nA: Not knowing the true facts, the defendant was not justified in using force to protect the woman because the man was privileged to recapture his briefcase.\nB: The defendant was justified in using any amount of force necessary in the woman's defense, since he believed she was in immediate danger.\nC: The defendant was not justified in using force, regardless of his belief about the situation, because the woman was attempting to steal the man's briefcase.\nD: The defendant was not justified in using force because he did not personally know the woman.\nE: Since the defendant did not stand in any personal relationship with the woman, he was not justified in using force in her defense.\nF: The defendant was justified in using force in the woman's defense, as he was acting in good faith and did not know the true circumstances.\nG: The defendant could use force in the woman's defense, provided that he did not cause more harm to the man than the woman was suffering.\nH: The defendant was justified in using reasonable force in the woman's defense, since his belief that she was in immediate danger of unlawful bodily harm from the man was both objectively and subjectively reasonable.\nI: The defendant was justified in using reasonable force in the woman's defense, since he reasonably believed she was in immediate danger of unlawful bodily harm from the man.\nJ: The defendant was justified in using force, even if the woman was in the wrong, because he believed she was in danger.\nPlease inference first, then provide the final answer (A/B/C/D/E/F/G/H/I/J) at the end, after 'Final Answer:'\n"""
        gold_label1 = \
"""B"""
        gold_label2 = \
"""A"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""The Fifth Amendment privilege against self-incrimination protects individuals from being compelled to testify against themselves, but it does not extend to corporations. The Supreme Court has consistently held that corporations do not have a Fifth Amendment privilege against self-incrimination. This principle was established in cases such as *Hale v. Henkel* (1906) and reaffirmed in *Braswell v. United States* (1988), where the Court ruled that corporate records must be produced even if they might be incriminating.

Additionally, the Fifth Amendment primarily protects testimonial evidence, not the production of preexisting corporate records. The act of producing documents may have a testimonial aspect under some circumstances, but that protection does not extend to corporate entities. The Supreme Court has held that corporations cannot assert the privilege against self-incrimination to refuse compliance with a subpoena for documents.

Given these principles, the best answer is **B: Yes, because a corporation has no Fifth Amendment privilege.** 

Final Answer: B"""
            a2 = \
"""Inference:

The alter ego rule states that a person who intervenes in defense of another stands in the shoes of the person they are defending. This means that if the person being defended had the right to use force in self-defense, then the intervenor also has that right. Conversely, if the person being defended was not legally entitled to use force, then the intervenor is also not justified in using force.

In this case:
1. The woman had just stolen the briefcase from the man.
2. The man was legally privileged to use force to recover his stolen property.
3. Since the woman was not legally entitled to use force against the man, the defendant, who was acting under a mistaken belief, was also not legally justified in using force.
4. The alter ego rule does not take into account the intervenor's reasonable but mistaken belief; it only considers whether the person being defended had the right to use force.

Now, considering the answer choices:
- **A** correctly states that the defendant was not justified in using force because the man had the right to recapture his briefcase.
- **B** is incorrect because justification is not based solely on the defendant’s belief but on whether the woman had the right to use force.
- **C** is correct in stating that the defendant was not justified, but it lacks an explanation related to the alter ego rule.
- **D** is incorrect because justification does not depend on knowing the person.
- **E** is incorrect because justification is not based solely on a personal relationship.
- **F** is incorrect because the alter ego rule does not consider good faith mistakes.
- **G** is incorrect because proportionality does not override the alter ego rule.
- **H** and **I** are incorrect because the alter ego rule does not justify force based solely on reasonable belief.
- **J** is incorrect because belief alone does not justify force when the person being defended was in the wrong.

Final Answer: A"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""The company cannot do that. The fifth Amendment privilege only apply to individuals. A company has no Fifth Amendment privilege.

Final Answer: B"""
            a2 = \
"""according to the Alter Ego Rule ego rule, a person who intervenes in defense of another stands in the shoes of the person they are defending. The girl is a theft who is doing illegal things, thus the man is not justify to use force to protect her. The man had right to recapture his briefcase.

Final Answer: A"""
    
    if task_name == 'ESNLI':
        q1 = \
"""We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is at a diner, ordering an omelette.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"""
        q3 = \
"""We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is outdoors, on a horse.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""neutral"""
        gold_label2 = \
"""contradiction"""
        gold_label3 = \
"""entailment"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Inference:

1. **Context Analysis**:
   - The context describes "A person on a horse jumps over a broken-down airplane." It provides a specific scenario of a horse and rider performing an action.

2. **Statement Analysis**:
   - The statement claims that "A person is training his horse for a competition." This introduces an assumption about the purpose of the action (training for a competition).

3. **Comparison**:
   - The context does not explicitly state or imply that the action is part of a competition or training for one. It only describes the act of jumping over a broken-down airplane.
   - The statement is plausible but not definitively supported by the given context. Additionally, it is not contradicted by the context, as training for a competition could reasonably involve such activities.

4. **Determination**:
   - The truth of the statement cannot be definitively determined based on the context provided. The context neither confirms nor denies the purpose of the activity.

---

Final Answer: Neutral"""
            a2 = \
"""Inference:

1. **Context Analysis**:
   The context describes a scene where a person on a horse is jumping over a broken-down airplane. This scenario provides specific visual and action-based information about the person, their location (likely outdoors near an airplane), and their activity (riding a horse and jumping).

2. **Statement Analysis**:
   The statement suggests that a person is at a diner, ordering an omelette. This describes a completely different scenario in terms of location (a diner vs. outdoors), activity (ordering food vs. riding a horse), and details (the presence of an airplane and horse vs. a diner and omelette).

3. **Comparison**:
   There is no overlap between the context and the statement, and the two scenarios are mutually exclusive based on the details provided. Since the context strongly implies the person's location and activity (outdoors, riding a horse), the statement suggesting they are at a diner becomes logically inconsistent with the context.

4. **Determination**:
   The statement is **definitely false** given the context because it directly contradicts the information provided.

---

Final Answer: Contradiction"""
            a3 = \
"""Inference:

1. **Context:** A person on a horse jumps over a broken-down airplane.
   - This implies that there is a person outdoors, riding a horse.
   - The scene described is consistent with being outdoors, as it involves a horse and a broken-down airplane, which are unlikely to be indoors.

2. **Statement:** A person is outdoors, on a horse.
   - This directly aligns with the information in the context. The context supports the idea that the person is outdoors and on a horse.

3. **Relationship:** The statement is fully supported by the context and does not introduce any conflicting or irrelevant information. It is definitely true based on the given context.

Final Answer: Entailment"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""The context 'a person on a horse jumps over a broken down airplane' does not mention if the person is triaining his horse. The person may not necessarily be riding his horse for competition training purpose. Therefore, the answer is neutral.

Final Answer: Neutral"""
            a2 = \
"""If a person is on a horse jumps over a airplane, then there is no way he is at dinner and ordering fooed. Therefore the answer is contradiction.

Final Answer: Contradiction.
"""
            a3 = \
"""If a person is on a horse jumps over a airplane, then he has to be outdoors, because airplanes are usually placed on outdoors. Therefore, the answer is entailment. 

Final Answer: Entailment.
"""

    if task_name == 'ECQA':
        q1 = \
"""We have the question: What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""1"""
        gold_label2 = \
"""3"""
        

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""To infer the correct answer, let us analyze the question and options step by step.

### Question:
"What might a person see at the scene of a brutal killing?"

This is asking about something that can be physically observed or perceived at such a scene. A "brutal killing" typically involves violence and bloodshed, leading to something visually evident.

### Analysis of Options:
1. **Bloody mess**: This aligns with what is commonly seen at a violent crime scene. Blood and disorder are realistic visual elements in such a scenario.
   
2. **Pleasure**: This is unrelated to what a person might visually see. While it could hypothetically relate to emotions, it is not something "seen."

3. **Being imprisoned**: Imprisonment is a consequence or action, not a visual element at the scene itself.

4. **Feeling of guilt**: Emotions, such as guilt, cannot be "seen." They are internal experiences, not physical observations.

5. **Cake**: A cake is unrelated to the context of a brutal killing and does not fit the scene's description.

### Conclusion:
The most reasonable and directly observable answer is **Option 1: bloody mess**.

### Final Answer: 1"""
            a2 = \
"""To infer the correct answer, let us analyze the question and each option:

### Question:
"If you want to set a romantic atmosphere you might light a candle where?"

### Analysis:
- **Romantic atmosphere** typically suggests intimacy, comfort, and settings associated with personal connections, like a private or cozy space.
- Lighting a candle is symbolic of creating mood lighting or ambiance, often associated with romance.

### Options:
1. **Dimly lit room**: This aligns with creating a romantic atmosphere. Candles complement a dimly lit room for a cozy and romantic setting.
2. **Synagogue**: While candles are lit in synagogues, it is not typically associated with creating a romantic atmosphere.
3. **Bedroom**: A bedroom is a common location for setting a romantic atmosphere. Candles are often used here for mood lighting.
4. **Birthday cake**: Candles on a birthday cake are for celebrations, not romance.
5. **Roses**: While roses symbolize romance, they are not a location to light a candle.

### Inference:
The two plausible options are **dimly lit room** and **bedroom**, as both could be associated with a romantic atmosphere. However, the **bedroom** is more specific and strongly associated with intimacy and romance.

---

Final Answer: 3"""
            
        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""A person may see the bloody scene of a brutal killing. For the first option, bloody mess matches bloody scene. For the second option, the word pleasure is not usually what a person see in brutal killing. For the third option, the scene about being imprisoned are not often related brutal killing. A person may be prisoned after killing the other people, but being imprisoned is the next step. For the fourth option, the person may have feeling of guilt after brutal killing, but there are often no killer are present at the scene of brutal killing, since he may run away.  For the fifth option, the scene of birthday cake are often not relevant to the scene of brutal killing. Therefore, the option is 1.

Final Answer: 1"""
            a2 = \
"""In a romantic atmosphere, people may want light a candle for some romantic things. The first option is a dimly lit room. This is often irrelevant to romantic things. The second option is synagogue. People may mirage in synagogue, but they may not light a candle there. The third option is bedroom, people might cheer their birthday in bedroom. In this case, people may light a candle inside the bedroom. The fourth option is birthday cake. People may very likely to light a candle on the birthday cake. The fifth option is roses. people do not light a candle on the roses. After analyzing all of the options, the most appropriate option would be 3.

Final Answer: 3"""

    if task_name == 'GSM8K':
        q1 = \
"""Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nFinal Answer: 72"""
        gold_label2 = \
"""Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\nFinal Answer: 10"""

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""In April, Natalia sold clips to 48 of her friends.  
In May, she sold **half as many clips as in April**, which is:

\[
\frac{48}{2} = 24
\]

The total number of clips Natalia sold in April and May is:

\[
48 + 24 = 72
\]

Final Answer: 72"""
            a2 = \
"""Weng earns $12 an hour, so her pay rate per minute is:

\[
\text{Hourly Rate} \div 60 = 12 \div 60 = 0.2 \text{ dollars per minute.}
\]

For 50 minutes of babysitting, her total earnings are:

\[
0.2 \times 50 = 10 \text{ dollars.}
\]

Final Answer: 10"""

        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Natalia sold 48 clips to her friends in April. She sold half as much which is 48/2 = <<48/2=24>>24 clips in May.\nIn total, Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\nFinal Answer: 72"""
            a2 = \
"""Weng earns $12/60minutes = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n\nFinal Answer: 10"""
    
    if 'MATH_ALGEBRA' in task_name.upper():
        q1 = \
"""Find the distance between the points (0,15) and (8,0).\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""What value of $x$ will give the minimum value for $x^2- 10x + 24$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""We use the distance formula: $$\\sqrt{(8 - 0)^2 + (0 - 15)^2} = \\sqrt{64 + 225} = \\boxed {17}.$$- OR -\n\nWe note that the points $(0, 15)$, $(8, 0)$, and $(0, 0)$ form a right triangle with legs of length 8 and 15. This is a Pythagorean triple, so the hypotenuse must have length $\\boxed{17}$."""
        gold_label2 = \
"""We start by completing the square.  \\[x^2-10x+24=(x-5)^2-1.\\] Since the square of a real number is at least 0, $(x-5)^2\\ge 0$ and $(x-5)^2-1 \\ge -1.$ Thus, the minimum value of the quadratic is $-1,$ which occurs when $x=\\boxed{5}."""
        
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""To find the distance between two points \((x_1, y_1)\) and \((x_2, y_2)\), we use the distance formula:

\[
\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]

For the points \((0, 15)\) and \((8, 0)\):
- \(x_1 = 0\), \(y_1 = 15\)
- \(x_2 = 8\), \(y_2 = 0\)

Substitute these values into the formula:

\[
\text{Distance} = \sqrt{(8 - 0)^2 + (0 - 15)^2}
\]
\[
\text{Distance} = \sqrt{8^2 + (-15)^2}
\]
\[
\text{Distance} = \sqrt{64 + 225}
\]
\[
\text{Distance} = \sqrt{289}
\]
\[
\text{Distance} = 17
\]

Final Answer: 17"""
            a2 = \
"""To find the value of \( x \) that minimizes the quadratic function \( x^2 - 10x + 24 \), we use the formula for the vertex of a parabola:

\[
x = -\frac{b}{2a}
\]

For the given quadratic \( x^2 - 10x + 24 \), the coefficients are:
- \( a = 1 \)
- \( b = -10 \)
- \( c = 24 \)

Substitute \( a \) and \( b \) into the formula:

\[
x = -\frac{-10}{2(1)} = \frac{10}{2} = 5
\]

The value of \( x \) that minimizes \( x^2 - 10x + 24 \) is:

Final Answer: 5"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""We use the distance formula to calculate the distance between the two points: $$\\sqrt{(8 - 0)^2 + (0 - 15)^2} = \\sqrt{64 + 225} = \\boxed {17}.$$\n\nFinal Answer: 17"""
            a2 = \
"""We start by completing the square.  \\[x^2-10x+24=(x-5)^2-1.\\] Since the square of a real number is at least 0, $(x-5)^2\\ge 0$ and $(x-5)^2-1 \\ge -1.$ Thus, the minimum value of the quadratic is $-1,$ which occurs when $x=\\boxed{5}.\n\nFinal Answer: 5"""
    
    if 'MATH_INTERMEDIATE_ALGEBRA' in task_name.upper():
        q1 = \
"""Given positive integers $x$ and $y$ such that $\\frac{1}{x} + \\frac{1}{2y} = \\frac{1}{7}$, what is the least possible value of $xy$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""The sum of the terms in an infinite geometric series is 15, and the sum of their squares is 45.  Find the first term.\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""Multiplying through by $14xy$, we have $14y + 7x = 2xy$, so $2xy - 7x - 14y = 0$. We then apply Simon's Favorite Factoring Trick by adding $49$ to both sides to get $2xy - 7x - 14y + 49 = 49$. We can then factor this to get $$(x-7)(2y-7) = 49$$Since $49$ factors to $7 \\cdot 7$ and $x$ and $y$ must be positive integers, the only possible solutions $(x,y)$ are $(8, 28), (14,7), \\text{and } (56,4)$. Out of these, $(14,7)$ yields the least possible value $xy$ of $\\boxed{98}$."""
        gold_label2 = \
"""Let $a$ be the first term, and let $r$ be the common ratio.  Then\n\\begin{align*}\n\\frac{a}{1 - r} &= 15, \\\\\n\\frac{a^2}{1 - r^2} &= 45.\n\\end{align*}From the first equation, $a = 15(1 - r).$  Substituting into the second equation, we get\n\\[\\frac{225 (1 - r)^2}{1 - r^2} = 45.\\]The denominator factors as $(1 + r)(1 - r),$ so the equation simplifies to\n\\[\\frac{5 (1 - r)}{1 + r} = 1.\\]Then $5 - 5r = 1 + r,$ so $r = \\frac{2}{3}.$  Then $a = 15 \\left( 1 - \\frac{2}{3} \\right) = \\boxed{5}.$"""
        
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""We are tasked with finding the least possible value of \( xy \) given the equation

\[
\frac{1}{x} + \frac{1}{2y} = \frac{1}{7},
\]

where \( x \) and \( y \) are positive integers. Let’s solve step by step.

---

### Step 1: Simplify the equation
Multiply through by \( 14xy \), the least common multiple of the denominators \( x, 2y, \) and \( 7 \), to eliminate the fractions:

\[
14y + 7x = 2xy.
\]

Rearrange this into a standard form:

\[
2xy - 14y - 7x = 0.
\]

---

### Step 2: Factorize
Factorize by grouping terms:

\[
x(2y - 7) = 14y.
\]

Divide through by \( y \) (since \( y > 0 \)):

\[
x = \frac{14y}{2y - 7}.
\]

---

### Step 3: Ensure \( x \) is a positive integer
For \( x \) to be a positive integer, \( \frac{14y}{2y - 7} \) must also be an integer. This means \( 2y - 7 \) must divide \( 14y \).

#### Check divisibility:
From the expression \( x = \frac{14y}{2y - 7} \), \( 2y - 7 \) must satisfy:
1. \( 2y - 7 > 0 \), so \( y > \frac{7}{2} \), meaning \( y \geq 4 \) (since \( y \) is an integer).
2. \( 2y - 7 \) divides \( 14y \).

Let \( k = 2y - 7 \). Then \( k \) divides \( 14y = 7k + 49 \), which simplifies to \( k \) dividing \( 49 \) (since \( 14y - 7k = 49 \)).

The divisors of \( 49 \) are \( 1, 7, 49 \).

---

### Step 4: Test possible values of \( k \)
For each divisor \( k \), calculate \( y \) and \( x \):

#### Case 1: \( k = 7 \)
\[
2y - 7 = 7 \implies 2y = 14 \implies y = 7.
\]

Substitute \( y = 7 \) into \( x = \frac{14y}{2y - 7} \):

\[
x = \frac{14 \cdot 7}{7} = 14.
\]

Thus, \( x = 14 \), \( y = 7 \), and \( xy = 14 \cdot 7 = 98 \).

#### Case 2: \( k = 49 \)
\[
2y - 7 = 49 \implies 2y = 56 \implies y = 28.
\]

Substitute \( y = 28 \) into \( x = \frac{14y}{2y - 7} \):

\[
x = \frac{14 \cdot 28}{49} = 8.
\]

Thus, \( x = 8 \), \( y = 28 \), and \( xy = 8 \cdot 28 = 224 \).

#### Case 3: \( k = 1 \)
\[
2y - 7 = 1 \implies 2y = 8 \implies y = 4.
\]

Substitute \( y = 4 \) into \( x = \frac{14y}{2y - 7} \):

\[
x = \frac{14 \cdot 4}{1} = 56.
\]

Thus, \( x = 56 \), \( y = 4 \), and \( xy = 56 \cdot 4 = 224 \).

---

### Step 5: Find the minimum \( xy \)
The possible values of \( xy \) are \( 98, 224, 224 \). The smallest value is:

\[
\boxed{98}.
\]

---

Final Answer: 98"""
            a2 = \
"""Let the first term of the geometric series be \( a \), and the common ratio be \( r \). The sum of the infinite geometric series is given by the formula:

\[
S = \frac{a}{1 - r}
\]

We are told that the sum of the series is 15, so:

\[
\frac{a}{1 - r} = 15
\]

Additionally, the sum of the squares of the terms in the series can also form a geometric series with the first term \( a^2 \) and common ratio \( r^2 \). The sum of the squares of the terms is given by:

\[
S_{\text{squares}} = \frac{a^2}{1 - r^2}
\]

We are told that the sum of the squares is 45, so:

\[
\frac{a^2}{1 - r^2} = 45
\]

Now we have two equations:

1. \( \frac{a}{1 - r} = 15 \)
2. \( \frac{a^2}{1 - r^2} = 45 \)

We can solve these equations step by step. Start with equation (1) to express \( a \) in terms of \( r \):

\[
a = 15(1 - r)
\]

Substitute this expression for \( a \) into equation (2):

\[
\frac{(15(1 - r))^2}{1 - r^2} = 45
\]

Simplify the left-hand side:

\[
\frac{225(1 - r)^2}{1 - r^2} = 45
\]

Now divide both sides by 45:

\[
\frac{5(1 - r)^2}{1 - r^2} = 1
\]

Expand both the numerator and denominator:

\[
\frac{5(1 - 2r + r^2)}{(1 - r)(1 + r)} = 1
\]

Simplify the equation:

\[
5(1 - 2r + r^2) = (1 - r)(1 + r)
\]

Now expand both sides:

\[
5(1 - 2r + r^2) = 1 - r^2
\]

Distribute the 5:

\[
5 - 10r + 5r^2 = 1 - r^2
\]

Combine like terms:

\[
5 - 10r + 5r^2 = 1 - r^2
\]

Move all terms to one side:

\[
5r^2 - 10r + 5 - 1 + r^2 = 0
\]

Simplify further:

\[
6r^2 - 10r + 4 = 0
\]

Now, divide through by 2:

\[
3r^2 - 5r + 2 = 0
\]

Solve this quadratic equation using the quadratic formula:

\[
r = \frac{-(-5) \pm \sqrt{(-5)^2 - 4(3)(2)}}{2(3)}
\]
\[
r = \frac{5 \pm \sqrt{25 - 24}}{6}
\]
\[
r = \frac{5 \pm 1}{6}
\]

Thus, \( r = \frac{6}{6} = 1 \) or \( r = \frac{4}{6} = \frac{2}{3} \).

Since \( r = 1 \) would not result in a converging series, we take \( r = \frac{2}{3} \).

Substitute \( r = \frac{2}{3} \) into the equation for \( a \):

\[
a = 15(1 - \frac{2}{3}) = 15 \times \frac{1}{3} = 5
\]

Thus, the first term of the series is \( \boxed{5} \).

Final Answer: 5"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Multiplying through by $14xy$, we have $14y + 7x = 2xy$, so $2xy - 7x - 14y = 0$. We then apply Simon's Favorite Factoring Trick by adding $49$ to both sides to get $2xy - 7x - 14y + 49 = 49$. We can then factor this to get $$(x-7)(2y-7) = 49$$Since $49$ factors to $7 \\cdot 7$ and $x$ and $y$ must be positive integers, the only possible solutions $(x,y)$ are $(8, 28), (14,7), \\text{and } (56,4)$. Out of these, $(14,7)$ yields the least possible value $xy$ of $\\boxed{98}$.\nFinal Answer: 98"""
            a2 = \
"""Let $a$ be the first term, and let $r$ be the common ratio.  Then\n\\begin{align*}\n\\frac{a}{1 - r} &= 15, \\\\\n\\frac{a^2}{1 - r^2} &= 45.\n\\end{align*}From the first equation, $a = 15(1 - r).$  Substituting into the second equation, we get\n\\[\\frac{225 (1 - r)^2}{1 - r^2} = 45.\\]The denominator factors as $(1 + r)(1 - r),$ so the equation simplifies to\n\\[\\frac{5 (1 - r)}{1 + r} = 1.\\]Then $5 - 5r = 1 + r,$ so $r = \\frac{2}{3}.$  Then $a = 15 \\left( 1 - \\frac{2}{3} \\right) = \\boxed{5}.$\nFinal Answer: 5"""

    if 'MATH_GEOMETRY' in task_name.upper():
        q1 = \
"""The rectangle with vertices $(-1, y), (7, y), (-1, 3)$, and $(7, 3)$ has an area of 72 square units, and $y$ is positive. What is the value of $y$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Three coplanar squares with sides of lengths two, four and six units, respectively, are arranged side-by-side, as shown so that one side of each square lies on line $AB$ and a segment connects the bottom left corner of the smallest square to the upper right corner of the largest square. What is the area of the shaded quadrilateral? [asy] size(150); defaultpen(linewidth(0.9)+fontsize(10));\nfill((2,0)--(6,0)--(6,3)--(2,1)--cycle,gray(0.8));\ndraw(scale(2)*unitsquare);\ndraw(shift(2,0)*scale(4)*unitsquare);\ndraw(shift(6,0)*scale(6)*unitsquare);\ndraw((0,0)--(12,6));\n\nreal d = 1.2; pair d2 = (0.9,0);\npair A = (-d,0), B = (12+d,0); dot(A,linewidth(3)); dot(B,linewidth(3)); label(\"A\",A,(0,-1.5)); label(\"B\",B,(0,-1.5)); draw(A-d2--B+d2,Arrows(4));\nlabel(\"2\",(1,2.7)); label(\"4\",(4,4.7)); label(\"6\",(9,6.7));\n[/asy]\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""First we write the rectangle's side lengths in terms of the coordinates provided. The length is $7-(-1)=8$ and the height is $y-3.$ It follows that $8(y-3)=72,$ and $y=\\boxed{12}.$ [asy]\nimport graph;\nsize(4cm);\ndefaultpen(linewidth(0.7)+fontsize(10));\ndotfactor=4;\nxaxis(Arrows(4));\nyaxis(ymin=-2,Arrows(4));\npair A=(-1,12), B=(7,12), C=(-1,3), D=(7,3);\npair[] dots = {A,B,C,D};\ndot(dots);\ndraw(A--B--D--C--cycle);\nlabel(\"$8$\",(A+B)/2,N);\nlabel(\"$y-3$\",(B+D)/2,E);\n[/asy]\nFinal Answer: 12"""
        gold_label2 = \
"""[asy]size(150); defaultpen(linewidth(0.9)+fontsize(10));\nfill((2,0)--(6,0)--(6,3)--(2,1)--cycle,gray(0.8));\ndraw(scale(2)*unitsquare);\ndraw(shift(2,0)*scale(4)*unitsquare);\ndraw(shift(6,0)*scale(6)*unitsquare);\ndraw((0,0)--(12,6));\n\nreal d = 1.2; pair d2 = (0.9,0);\npair A = (-d,0), B = (12+d,0); dot(A,linewidth(3)); dot(B,linewidth(3)); label(\"A\",A,(0,-1.5)); label(\"B\",B,(0,-1.5)); draw(A-d2--B+d2,Arrows(4));\nlabel(\"2\",(1,2.7)); label(\"4\",(4,4.7)); label(\"6\",(9,6.7)); label(\"6\",(12.7,3)); label(\"3\",(6.7,1.5)); label(\"1\",(2.5,0.5)); label(\"$2$\",(1,-0.7)); label(\"$4$\",(4,-0.7)); label(\"$6$\",(9,-0.7));\n[/asy] Consider the three right triangles $T_1, T_2, T_3$ formed by the line $AB$, the segment connecting the bottom left corner of the smallest square to the upper right corner of the largest square, and a side of the smallest, medium, and largest squares, respectively. Since all three triangles share an angle, it follows that they must be similar. Notice that the base of $T_3$ is equal to $2+4+6 = 12$, and its height is equal to $6$. This, the height-to-base ratio of each of $T_1$ and $T_2$ is equal to $6/12 = 1/2$. Since the base of $T_1$ is $2$ and the base of $T_2$ is $2+4 = 6$, it follows that their heights are, respectively, $2 \\cdot (1/2) = 1$ and $6 \\cdot (1/2) = 3$. The shaded region is a trapezoid with bases $1$ and $3$ and altitude $4$, and area $\\frac{4(1+3)}{2} = \\boxed{8}$.\nFinal Answer: 8"""
        
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""To find the value of \( y \), let us calculate the area of the rectangle based on its vertices.

### Step 1: Determine the dimensions of the rectangle
The given vertices are:
- \( (-1, y) \) and \( (7, y) \): These two points lie on the same horizontal line, so the length of the rectangle is the horizontal distance between these two points. The length is:
\[
7 - (-1) = 7 + 1 = 8.
\]

- \( (-1, y) \) and \( (-1, 3) \): These two points lie on the same vertical line, so the width of the rectangle is the vertical distance between \( y \) and 3. The width is:
\[
|y - 3|.
\]

### Step 2: Use the area formula
The area of a rectangle is given by:
\[
\text{Area} = \text{Length} \times \text{Width}.
\]
Substitute the known values:
\[
72 = 8 \cdot |y - 3|.
\]

### Step 3: Solve for \( |y - 3| \)
Divide both sides by 8:
\[
|y - 3| = \frac{72}{8} = 9.
\]

### Step 4: Solve for \( y \)
The absolute value equation \( |y - 3| = 9 \) implies two cases:
1. \( y - 3 = 9 \), which gives:
\[
y = 12.
\]
2. \( y - 3 = -9 \), which gives:
\[
y = -6.
\]

### Step 5: Choose the valid solution
Since \( y \) is given as positive, we select \( y = 12 \).

Final Answer: 12"""
            a2 = \
"""To determine the area of the shaded quadrilateral, we proceed step by step.

---

### Step 1: Understanding the problem
We are given three squares arranged side by side:
- A square with side length \( 2 \) is positioned from \( x = 0 \) to \( x = 2 \),
- A square with side length \( 4 \) is positioned from \( x = 2 \) to \( x = 6 \),
- A square with side length \( 6 \) is positioned from \( x = 6 \) to \( x = 12 \).

A diagonal line connects the bottom-left corner of the smallest square at \( (0, 0) \) to the top-right corner of the largest square at \( (12, 6) \).

The shaded quadrilateral lies between this diagonal line and the horizontal line \( AB \), spanning from \( x = 2 \) to \( x = 6 \) (the middle square).

---

### Step 2: Equation of the diagonal line
The diagonal line passes through \( (0, 0) \) and \( (12, 6) \). Its slope is:
\[
m = \frac{6 - 0}{12 - 0} = \frac{1}{2}.
\]
Thus, the equation of the diagonal line is:
\[
y = \frac{1}{2}x.
\]

---

### Step 3: Vertices of the shaded quadrilateral
The shaded quadrilateral is bounded by:
- The diagonal line \( y = \frac{1}{2}x \),
- The horizontal line \( AB \) (\( y = 0 \)),
- Vertical lines at \( x = 2 \) and \( x = 6 \).

#### Finding intersection points:
- At \( x = 2 \), \( y = \frac{1}{2}(2) = 1 \), so the intersection point is \( (2, 1) \).
- At \( x = 6 \), \( y = \frac{1}{2}(6) = 3 \), so the intersection point is \( (6, 3) \).

The vertices of the shaded quadrilateral are:
\[
(2, 0), \, (2, 1), \, (6, 3), \, (6, 0).
\]

---

### Step 4: Area of the shaded quadrilateral
To find the area of the quadrilateral, we use the shoelace formula. Denoting the vertices in order as \( (x_1, y_1), (x_2, y_2), \dots, (x_4, y_4) \), the area is:
\[
\text{Area} = \frac{1}{2} \left| x_1y_2 + x_2y_3 + x_3y_4 + x_4y_1 - (y_1x_2 + y_2x_3 + y_3x_4 + y_4x_1) \right|.
\]

Substitute the vertices \( (2, 0), (2, 1), (6, 3), (6, 0) \):
\[
\text{Area} = \frac{1}{2} \left| 2 \cdot 1 + 2 \cdot 3 + 6 \cdot 0 + 6 \cdot 0 - (0 \cdot 2 + 1 \cdot 6 + 3 \cdot 6 + 0 \cdot 2) \right|.
\]
Simplify:
\[
\text{Area} = \frac{1}{2} \left| 2 + 6 + 0 + 0 - (0 + 6 + 18 + 0) \right|.
\]
\[
\text{Area} = \frac{1}{2} \left| 8 - 24 \right| = \frac{1}{2} \cdot 16 = 8.
\]

---

Final Answer: 8"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""First we write the rectangle's side lengths in terms of the coordinates provided. The length is $7-(-1)=8$ and the height is $y-3.$ It follows that $8(y-3)=72,$ and $y=\\boxed{12}.$ [asy]\nimport graph;\nsize(4cm);\ndefaultpen(linewidth(0.7)+fontsize(10));\ndotfactor=4;\nxaxis(Arrows(4));\nyaxis(ymin=-2,Arrows(4));\npair A=(-1,12), B=(7,12), C=(-1,3), D=(7,3);\npair[] dots = {A,B,C,D};\ndot(dots);\ndraw(A--B--D--C--cycle);\nlabel(\"$8$\",(A+B)/2,N);\nlabel(\"$y-3$\",(B+D)/2,E);\n[/asy]\nFinal Answer: 12"""
            a2 = \
"""[asy]size(150); defaultpen(linewidth(0.9)+fontsize(10));\nfill((2,0)--(6,0)--(6,3)--(2,1)--cycle,gray(0.8));\ndraw(scale(2)*unitsquare);\ndraw(shift(2,0)*scale(4)*unitsquare);\ndraw(shift(6,0)*scale(6)*unitsquare);\ndraw((0,0)--(12,6));\n\nreal d = 1.2; pair d2 = (0.9,0);\npair A = (-d,0), B = (12+d,0); dot(A,linewidth(3)); dot(B,linewidth(3)); label(\"A\",A,(0,-1.5)); label(\"B\",B,(0,-1.5)); draw(A-d2--B+d2,Arrows(4));\nlabel(\"2\",(1,2.7)); label(\"4\",(4,4.7)); label(\"6\",(9,6.7)); label(\"6\",(12.7,3)); label(\"3\",(6.7,1.5)); label(\"1\",(2.5,0.5)); label(\"$2$\",(1,-0.7)); label(\"$4$\",(4,-0.7)); label(\"$6$\",(9,-0.7));\n[/asy] Consider the three right triangles $T_1, T_2, T_3$ formed by the line $AB$, the segment connecting the bottom left corner of the smallest square to the upper right corner of the largest square, and a side of the smallest, medium, and largest squares, respectively. Since all three triangles share an angle, it follows that they must be similar. Notice that the base of $T_3$ is equal to $2+4+6 = 12$, and its height is equal to $6$. This, the height-to-base ratio of each of $T_1$ and $T_2$ is equal to $6/12 = 1/2$. Since the base of $T_1$ is $2$ and the base of $T_2$ is $2+4 = 6$, it follows that their heights are, respectively, $2 \\cdot (1/2) = 1$ and $6 \\cdot (1/2) = 3$. The shaded region is a trapezoid with bases $1$ and $3$ and altitude $4$, and area $\\frac{4(1+3)}{2} = \\boxed{8}$.\nFinal Answer: 8"""

    if 'SQUAD' in task_name.upper():
        q1 = \
"""Given the question: When did Beyonce start becoming popular? and the context: Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (text span) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the question: What areas did Beyonce compete in when she was growing up? and the context: Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (text span) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""In the late 1990s"""
        gold_label2 = \
"""singing and dancing"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

1. **Key Question:** When did Beyoncé start becoming popular?
- The question focuses on identifying the period when Beyoncé's fame began to rise.

2. **Context Analysis:**
- Beyoncé was born in 1981.
- As a child, she participated in singing and dancing competitions, which hints at early engagement in performance but not widespread popularity.
- She rose to fame in the **late 1990s** as the lead singer of Destiny's Child. This marks the beginning of her prominence in the music industry.
- Her solo career began with the release of her debut album *Dangerously in Love* in 2003, solidifying her status as a solo artist but occurring after her initial rise with Destiny's Child.

3. **Inference:**
- Beyoncé started becoming popular in the late 1990s due to her role as the lead singer of Destiny's Child.

### Final Answer: In the late 1990s."""
            a2 = \
"""### Inference:

The question asks about the areas in which Beyoncé competed during her childhood. The context specifies that Beyoncé, as a child, participated in **singing** and **dancing**. These are explicitly mentioned in the context and directly answer the question.

---

### Final Answer: singing and dancing"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""The context mentions that 'rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child.’, so Beyoncé did start becoming popular 'In the late 1990s'.

Final Answer: In the late 1990s"""

            a2 = \
"""The context mentions that 'she performed in various singing and dancing competitions as a child’, so the area is 'singing and dancing’.

Final Answer: singing and dancing"""
    
    if 'DROP' in task_name.upper():
        q1 = \
"""Given the passage: To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007\nPlease answer the question: Who caught the shortest touchdown pass?\n\nPlease inference first, then provide the final answer at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the passage: To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass.  The Lions won their regular season opener for the first time since 2007\nPlease answer the question: How many field goals did the Lions score?\n\nPlease inference first, then provide the final answer at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""Calvin Johnson"""
        gold_label2 = \
"""2"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

1. The passage describes a football game between the Detroit Lions and the Tampa Bay Buccaneers.
2. Several touchdowns are mentioned, along with their yardage:
   - Calvin Johnson caught a **36-yard touchdown pass** in the second quarter.
   - Tony Scheffler caught an **11-yard touchdown pass** in the second quarter.
   - Calvin Johnson caught a **1-yard touchdown pass** in the third quarter.
   - Mike Williams caught a **5-yard touchdown pass** in the game's final moments.
3. Among the touchdown passes, the shortest is **1 yard**, caught by Calvin Johnson in the third quarter.

Final Answer: Calvin Johnson"""
            a2 = \
"""### Inference:
The passage describes the scoring events of a football game between the Detroit Lions and the Tampa Bay Buccaneers. Specifically, it mentions field goals as part of the Lions' scoring:

1. In the first quarter, the Lions scored a **23-yard field goal** by Jason Hanson.  
2. Later in the first quarter, they responded with a **28-yard field goal**.

No additional field goals by the Lions are mentioned in the passage.

Final Answer: 2"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""We could summary the touchdown pass mentioned by the passage. According to the passage, we know that 
Calvin Johnson caught a **36-yard touchdown pass** in the second quarter.
Tony Scheffler caught an **11-yard touchdown pass** in the second quarter.
Calvin Johnson caught a **1-yard touchdown pass** in the third quarter.
Mike Williams caught a **5-yard touchdown pass** in the game's final moments.
Thus Calvin Johnson caught the shortest touchdown pass which is 1-yard touch down pass.

Final Answer: Calvin Johnson"""

            a2 = \
"""By reading the passage, we can count the number of field goals did the Lions score. In the first quarter, 'The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson.' Then, 'The Lions responded with a 28-yard field goal.', which is also in the first quarter. There is no other field goal scored by the Lions we can find from the passage.

Final Answer: 2"""
    
    if 'WINOGRANDE' in task_name.upper():
        q1 = \
"""Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""2"""
        gold_label2 = \
"""1"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Let's analyze the question step by step:

### Statement Analysis:
- **"Ian volunteered to eat Dennis's menudo after already having a bowl"**: This implies that Ian is taking an active role in eating more menudo, which suggests he has no issue eating it.
- **"because _ despised eating intestine"**: The blank refers to someone who does not like eating intestine.

### Logical Deduction:
- If Ian is willing to eat Dennis's menudo even after having a bowl himself, it indicates that Ian is fine with eating menudo (which contains intestine).
- Therefore, the person who despises eating intestine must be Dennis, as he is the one whose menudo Ian is eating.

Final Answer: 2"""
            a2 = \
"""### Inference:

1. The sentence states that Ian volunteered to eat Dennis's menudo after already having a bowl. This implies Ian had already eaten one serving and is now taking another, which suggests Ian enjoys eating menudo.
2. The phrase "because _ enjoyed eating intestine" indicates the subject of the sentence is someone who likes menudo, which is made from intestines.
3. Since Ian is the one eating a second serving, it is logical to infer that Ian is the one who enjoys eating intestine.

Final Answer: 1"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Ian volunteered to eat Dennis's menudo after having a bowl, because someone does not like intestine. Since he already having a bowl, so he may feel full. he volunteered to eat so as to help someone who might not want to eat intestine. Thus Dennis may be the one who does not want to eat intestine.

Final Answer: 2"""
            a2 = \
"""Ian volunteered to eat Dennis's menudo after having a bowl, because someone like intestine. In this situation, Ian should be the one who like intestine.

Final Answer: 1"""
    
    if 'PIQA' in task_name.upper():
        q1 = \
"""Given the question: When boiling butter, when it's ready, you can\n\nWhat option is correct?\nOption 1: Pour it onto a plate\nOption 2: Pour it into a jar\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the question: To permanently attach metal legs to a chair, you can\n\nWhat option is correct?\nOption 1: Weld the metal together to get it to stay firmly in place\nOption 2: Nail the metal together to get it to stay firmly in place\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""2"""
        gold_label2 = \
"""1"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""To infer the correct answer, let's consider the context of boiling butter. When butter is boiled, the solids separate, leaving behind clarified butter or ghee. This liquid is usually stored for later use. Pouring it onto a plate wouldn't be practical because the clarified butter needs containment to stay usable. Hence, pouring it into a jar for storage is the most logical option.

Final Answer: 2"""
            a2 = \
"""### Inference:

- **Welding**: Welding is a common and effective method to permanently join metal parts. It involves melting the metal surfaces and fusing them together, creating a strong and durable bond. This method is widely used for structural applications and is considered permanent.

- **Nailing**: Nailing is not a practical or reliable method for attaching metal components. Nails are primarily designed for wood and other softer materials. Metal would resist penetration by nails and, even if forced, the bond would be weak and not considered permanent.

Final Answer: 1"""
        
        if varient_name == 'openai_human_written_examples':
            a1 = \
"""people usually store boiled butter(which is liquid) in jar instead of plate.

Final Answer: 2"""
            a2 = \
"""If we nail the metal together, it may become loose after a while. However, if we meld the metal together, it will stay permanently.

Final Answer: 1"""
        
    if 'MMLU' == task_name.upper():
        q1 = \
"""Given the question: One afternoon, a pilot was flying a small airplane when it suddenly ran out of gas. As he was coming in for an emergency landing, the plane crossed into a neighboring state at a very low altitude. At this time, a 9-year-old boy was walking to school when he was struck and injured by an object, which may have fallen from the plane. In federal court, a negligence suit was brought against the pilot by the father of the boy for his son. Accompanied by his father, the boy had visited an attorney for preliminary discussions regarding the case. However, the father did not retain the attorney to represent his son in the lawsuit. Instead, the father hired another lawyer to handle the case. At trial, the pilot's attorney calls the consulting attorney to testify what the boy had said to him regarding his physical condition during the consultation that the attorney had had with the boy and his father. The attorney's testimony is\n\nand the options:\nA: admissible, because the attorney-client privilege was waived by the filing of the lawsuit.\nB: admissible, because there is no privilege of confidentiality when a person other than the client is present at the attorney-client consultation.\nC: inadmissible, because the attorney-client privilege prevents such a breach of confidential communications.\nD: inadmissible, because it was a statement of physical condition not made for the purpose of obtaining medical treatment.\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the question: A resident announced his candidacy for state representative. A law in the state requires new political entrants (regardless of party affiliation) to obtain three times the number of signatures as other candidates who have run for office previously. The resident, however, failed to obtain the necessary number of authenticating signatures to have his name placed on the ballot. The resident filed a complaint in federal district court alleging the unconstitutionality of the authenticating requirement. Which of the following, if established, is the state's strongest argument for sustaining the validity of the authenticating requirement?\n\nand the options:\nA: The authenticating requirement was necessary to further a compelling state interest.\nB: A similar authenticating statute was held to be constitutional in another state the previous year.\nC: The resident's petition contained a large number of false signatures.\nD: Two other candidates had successfully petitioned to have their names included on the ballot.\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""C"""
        gold_label2 = \
"""A"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

1. **Attorney-Client Privilege**: This privilege generally protects confidential communications between an attorney and a client made for the purpose of seeking legal advice. It applies to both the client and the client's attorney and can only be waived by the client.

2. **Presence of a Third Party**: The presence of a third party (in this case, the father) during an attorney-client communication can potentially negate the confidentiality requirement, thus nullifying the privilege. However, this depends on whether the third party was necessary for the consultation.

3. **Filing of the Lawsuit**: Filing a lawsuit does not automatically waive the attorney-client privilege unless the client’s confidential communications are directly placed at issue in the case (e.g., by asserting a claim that requires disclosure of privileged communications).

4. **Purpose of the Statement**: Statements about physical condition are typically considered confidential if made during an attorney-client consultation. However, if the purpose of the statement was not to seek medical treatment, it might not be protected by other privileges, such as the physician-patient privilege.

---

### Option Analysis:

- **A: Admissible, because the attorney-client privilege was waived by the filing of the lawsuit.**  
  Incorrect. Filing the lawsuit does not automatically waive the attorney-client privilege, particularly when the communications are unrelated to the legal strategy or the issue at trial.

- **B: Admissible, because there is no privilege of confidentiality when a person other than the client is present at the attorney-client consultation.**  
  Incorrect. The presence of the father does not necessarily negate confidentiality, as the father is the legal guardian and may be considered necessary for the consultation due to the boy’s age.

- **C: Inadmissible, because the attorney-client privilege prevents such a breach of confidential communications.**  
  Correct. The communication was made during an attorney-client consultation, and the privilege applies because the father’s presence does not nullify confidentiality, given the context of the boy’s age and the father’s role as the boy’s guardian.

- **D: Inadmissible, because it was a statement of physical condition not made for the purpose of obtaining medical treatment.**  
  Incorrect. This focuses on the wrong privilege (physician-patient privilege) rather than attorney-client privilege, which is the central issue.

---

### Final Answer: C"""
            a2 = \
"""To answer this question, let us analyze the options based on the legal reasoning principles that would apply to an alleged unconstitutional law.

### Question Breakdown:
The resident is challenging a state law requiring new political entrants to collect a greater number of signatures than other candidates. The argument must justify the constitutionality of this requirement under the federal court's scrutiny.

Federal courts often analyze such laws under the **Equal Protection Clause** or the **First Amendment**, considering whether the law serves a **compelling state interest** and whether it is **narrowly tailored** to achieve that interest without unduly burdening political participation.

### Evaluation of Options:

**A: The authenticating requirement was necessary to further a compelling state interest.**  
This directly addresses the constitutional test applied by federal courts. For a law to survive scrutiny, especially one imposing differential burdens on new entrants, the state must demonstrate that the law serves a compelling state interest and is appropriately tailored to achieve that interest. This is the strongest argument to justify the law.

**B: A similar authenticating statute was held to be constitutional in another state the previous year.**  
Although precedents in other jurisdictions may be persuasive, the constitutionality of the law depends on its specific application in this case and whether it meets constitutional scrutiny. A decision from another state is not binding and offers limited support.

**C: The resident's petition contained a large number of false signatures.**  
This fact pertains to the resident's failure to meet the existing legal requirement but does not address the constitutionality of the requirement itself. Therefore, it does not strengthen the state's argument for the validity of the law.

**D: Two other candidates had successfully petitioned to have their names included on the ballot.**  
While this shows that the requirement is not impossible to meet, it does not directly address whether the law itself is constitutional. The success of other candidates does not justify a law that imposes an unequal burden on new entrants.

### Conclusion:
The strongest argument for sustaining the validity of the authenticating requirement is that it is **necessary to further a compelling state interest**, as this aligns with the legal framework used to evaluate such claims.

Final Answer: A"""

        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Attorney-Client Privilege protects confidential communications between a client and an attorney made for the purpose of obtaining legal advice. It is only waived if the client (or their representative) takes action that explicitly or implicitly forfeits this protection.

Option A: Incorrect. Filing a lawsuit does not waive attorney-client privilege. Privilege applies unless the privilege-holder (the boy or his father) explicitly waives it.
Option B: Incorrect. The father’s presence does not eliminate the privilege, as he is a necessary participant in the consultation due to the boy's status as a minor.
Option C: Correct. The attorney-client privilege prevents the consulting attorney from testifying about confidential communications during the consultation.
Option D: Incorrect. The reason for the statement (not for medical treatment) does not override the protection of attorney-client privilege.

Final Answer: C"""

            a2 = \
"""### Question Breakdown:
The resident is challenging a state law requiring new political entrants to collect a greater number of signatures than other candidates. The argument must justify the constitutionality of this requirement under the federal court's scrutiny.

Federal courts often analyze such laws under the **Equal Protection Clause** or the **First Amendment**, considering whether the law serves a **compelling state interest** and whether it is **narrowly tailored** to achieve that interest without unduly burdening political participation.

### Evaluation of Options:

**A: The authenticating requirement was necessary to further a compelling state interest.**  
This directly addresses the constitutional test applied by federal courts. For a law to survive scrutiny, especially one imposing differential burdens on new entrants, the state must demonstrate that the law serves a compelling state interest and is appropriately tailored to achieve that interest. This is the strongest argument to justify the law.

**B: A similar authenticating statute was held to be constitutional in another state the previous year.**  
The constitutionality of the law depends on its specific application in this case and whether it meets constitutional scrutiny. A decision from another state is not binding and offers limited support.

**C: The resident's petition contained a large number of false signatures.**  
This fact pertains to the resident's failure to meet the existing legal requirement but does not address the constitutionality of the requirement itself. Therefore, it does not strengthen the state's argument for the validity of the law.

**D: Two other candidates had successfully petitioned to have their names included on the ballot.**  
It does not directly address whether the law itself is constitutional. The success of other candidates does not justify a law that imposes an unequal burden on new entrants.

Final Answer: A"""
    
    if 'MMLU_MORAL_SCENARIOS' == task_name.upper():
        q1 = \
"""Given the question: For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I didn't think his wife was even remotely attractive, so I decided I would never ask her out even if she was unmarried. Scenario 2 | I chased my sister with a hot iron.\n\nand the options:\nA: Wrong, Wrong\nB: Wrong, Not wrong\nC: Not wrong, Wrong\nD: Not wrong, Not wrong\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the question: For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | When the black family moved onto my block, I was worried about property values. Scenario 2 | I let a rabid dog loose in the nursing home.\n\nand the options:\nA: Wrong, Wrong\nB: Wrong, Not wrong\nC: Not wrong, Wrong\nD: Not wrong, Not wrong\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""C"""
        gold_label2 = \
"""A"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""Let's analyze the moral implications of each scenario according to ordinary moral standards in the U.S. as of 2020.

Scenario 1:
"I didn't think his wife was even remotely attractive, so I decided I would never ask her out even if she was unmarried."

- The person is expressing their personal opinion about someone's attractiveness and making a private decision not to pursue a relationship.
- There is no action that causes harm or violates moral norms.
- In general, personal preferences in attraction and decisions based on them are not considered morally wrong.
- **Conclusion:** Not morally wrong.

Scenario 2:
"I chased my sister with a hot iron."

- Chasing someone with a hot iron is an act that could potentially cause harm or at least instill fear.
- This could be classified as aggressive, threatening, or even abusive behavior, depending on the context.
- Such actions are generally considered morally wrong by ordinary moral standards, especially if they involve the potential for physical harm.
- **Conclusion:** Morally wrong.

Thus, Scenario 1 is **not morally wrong**, and Scenario 2 is **morally wrong**.

Final Answer: C"""

            a2 = \
"""Let's analyze the moral implications of each scenario according to ordinary moral standards in the U.S. as of 2020.

Scenario 1:
*"When the black family moved onto my block, I was worried about property values."*

- This statement expresses a racial bias, which is widely considered morally wrong in the U.S. as of 2020.
- The concern about property values based solely on a family's race is rooted in discriminatory beliefs that have historically contributed to housing segregation and systemic racism.
- While this statement does not indicate any direct harmful action, the expressed attitude itself reflects a form of racial prejudice.
- **Conclusion:** Morally wrong.

Scenario 2:
*"I let a rabid dog loose in the nursing home."*

- Letting a rabid dog loose in a nursing home is an act that endangers vulnerable people.
- It is highly irresponsible and malicious, as rabies is a deadly disease, and this action could lead to severe harm or death.
- Ordinary moral standards in the U.S. would clearly consider this to be a severe moral wrongdoing.
- **Conclusion:** Morally wrong.

Since both scenarios involve something considered morally wrong, the correct answer is:

Final Answer: A"""

        if varient_name == 'openai_human_written_examples':
            a1 = \
"""Thinking someone is not attractive and decide to not to pursue a relationship with her is not morally wrong. that is just his personal decision. he is not harmming anyone. 

Chasing someone with hot iron is very dangerous. it makes other people unsafe. it is wrong.

Final Answer: C"""

            a2 = \
"""Worry about property values when the black family moved onto his own block is racist. This is definitely morally wrong.

Let a rabid dog loose in the nursing home is very dangerous. the dog might harm other people. This is morally wrong.

Final Answer: A"""

    if 'AGIEVAL' in task_name.upper():
        q1 = \
"""Given the statement: In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.\n\nand the question: Based on the above statement, which of the following can be derived?\n\nand the options:\n(A)Civic Park is north of the administrative service area\n(B)The leisure area is southwest of the cultural area\n(C)The cultural district is in the northeast of the business district\n(D)The business district is southeast of the leisure area\n\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        q2 = \
"""Given the statement: The company sent three young staff members to the South for business trip.The three of them happened to be sitting in a row.At least one of the two people sitting on the right side of 24 years old was 20 years old.20 years old.At least one of the two people sitting on the left of the accountant is a salesperson, and one of the two people sitting on the right of the salesperson is also a salesperson.\n\nand the question: So what are the three young people on business?\n\nand the options:\n(A)20-year-old accountant, 20-year-old salesperson, 24-year-old salesperson\n(B)20-year-old accountant, 24-year-old salesperson, 24-year-old salesperson\n(C)24-year-old accountant, 20-year-old salesperson, 20-year-old salesperson\n(D)20-year-old accountant, 20-year-old accountant, 24-year-old salesperson\n\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n"""
        
        gold_label1 = \
"""A"""
        gold_label2 = \
"""A"""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""### Inference:

1. **Community placement around the Civic Park:**
- Communities are centered on the Civic Park and include:
- Southeast: Cultural Area
- Northwest: Leisure Area
- Commercial Area
- Administrative Service Area

The exact placement of the Commercial and Administrative Service Areas isn't explicitly stated relative to the Civic Park.

2. **Relationship between specific areas:**
- The Administrative Service Area is southwest of the Cultural Area.
- The Cultural Area is southeast of the Leisure Area.

This indicates:
- The Administrative Service Area is located in the south or southwest of the park.
- The Cultural Area is in the southeast.
- The Leisure Area is northwest of the Cultural Area.

3. **Relative position of other areas:**
- By logical inference:
- The Administrative Service Area lies in the southwest of the Civic Park.
- The Cultural Area, being in the southeast, is northeast of the Administrative Service Area.
- The Leisure Area lies northwest of the Civic Park and is also northwest of the Cultural Area.
- The Commercial Area's exact position isn't provided, but it might fit the northeast or central area.

4. **Evaluating the options:**
- (A) **Civic Park is north of the Administrative Service Area:**
- The Civic Park is indeed north of the southwest, where the Administrative Service Area is located. This statement is true.

- (B) **The Leisure Area is southwest of the Cultural Area:**
- The Leisure Area is northwest of the Cultural Area, not southwest. This statement is false.

- (C) **The Cultural District is in the northeast of the Business District:**
- The location of the Business District is not provided, so this statement cannot be inferred as true.

- (D) **The Business District is southeast of the Leisure Area:**
- Based on logical placement, this statement aligns with the description but cannot be confirmed without explicit evidence about the Business District.

### Final Answer: A"""
            a2 = \
"""### Inference:

#### Given Statements:
1. **The three young people are sitting in a row.**
2. **At least one of the two people sitting on the right of the 24-year-old is 20 years old.**
- This implies that if the 24-year-old is not at the far right, at least one of the two people to their right is 20 years old.
3. **At least one of the two people sitting on the left of the accountant is a salesperson.**
- This means the accountant is not at the far left, and one of the two people to their left must be a salesperson.
4. **One of the two people sitting on the right of the salesperson is also a salesperson.**
- This implies there must be at least one salesperson to the right of another salesperson.

#### Analysis:
1. **Arrangement Possibilities:**
- The 24-year-old cannot be at the far right because there wouldn’t be two people on their right to satisfy Statement 2. Hence, the 24-year-old must be in the middle or on the left.
- The accountant cannot be at the far left because Statement 3 would be violated.

2. **Age and Occupation Distribution:**
- The ages are 20, 20, and 24.
- The occupations are accountant, salesperson, and salesperson.

3. **Checking Conditions:**
- If the 24-year-old is in the middle:
- The people on the left and right must include a 20-year-old each to satisfy Statement 2.
- The accountant cannot be at the far right due to Statement 3. Therefore, the accountant must be at the far left, and one of the two on their left must be a salesperson.
- If the 24-year-old is on the left:
- The conditions regarding ages and roles become inconsistent because there must be two people to their right, one of whom is a 20-year-old, and the accountant cannot be on the far right.

#### Valid Configurations:
- **Configuration A** (20-year-old accountant, 24-year-old salesperson, 20-year-old salesperson):
- Accountant at far left satisfies Statement 3 (there's a salesperson to their left).
- The 24-year-old salesperson in the middle satisfies Statement 2 (the person on the right is 20 years old).
- The salesperson condition (one salesperson on the right of another salesperson) is also satisfied.

#### Verification of Options:
- **Option A:** 20-year-old accountant, 20-year-old salesperson, 24-year-old salesperson.
- Matches the above valid configuration.
- **Option B:** Violates Statement 3 (the accountant must have a salesperson on their left).
- **Option C:** Violates Statement 3 (accountant cannot be at far left without a salesperson to their left).
- **Option D:** Violates Statement 4 (salesperson arrangement inconsistent).

### Final Answer: A"""

        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""To answer the question, the first step is to analyze the location of each community given the description in the question. According to the question, we know that the cultural area is southeast of the leisure area. Since the condition mention that the special communities are located in the southeast, northwest, centered on the citizen park. According to those information, the administrative area is likely placed at the southwest relative to the center area and the leisure area is likely to placed at the northeast(NW) of the center area. According to another condition, we know that the administrative service area is southwest of the cultural area. Since we already assume cultural area is placed at the northeast(NE) of the center area, so the administrative service area is placed at the south west(SW) of the cultural area. we have another communities, which is commercial area. since we already have
cultural area(NE)
administrative service area(SW)
northeast(NW)

so the commercial area is likely placed at southeast.
so we have

commercial area(SE)
cultural area(NE)
administrative service area(SW)
leisure area(NW)

According to this condition, we can inference that
(A) Since administrative service area is at the south west relative to the citizen park, which is also civic park, A is correct.
(B) According to the relative position that we have inferenced, leisure area is at NW and cultural area is at NE, the leisure area should be at west relative to cultural area. It is hard to say if the leisure are is at north or south relative to the cultural area.
(C) According to the relative position that we have inferenced, we have commercial area(SE) and cultural area(NE). In other words, cultural area shoul be on the north side of the commercial(business) area. however, we cannot determine if cultural area is on the left or the right side of the commercial area.
(D) According to the relative position that we have inference, we have commercial area(SE) and leisure area(NW). so the business(commercial) area should be on the south east side of the leisure area.

Therefore, only (A) is correct.

Final Answer: A"""

            a2 = \
"""According to the condition, we know that at least one of the two people sitting on the left side of the accountant is a salesperson  so the potential options are:
left        middle       right
(2) ?, salesperson, accountant
(3) salesperson, ?, accountant

We also know that and at least one of the two people sitting on the right of the salesperson is also a salesperson, so we have
left         middle      right
salesperson, salesperson, accountant

we know that at least one 20 year old person sitting on the right side of the 24 years old person. In other words, the 24 years old person has to be on the middle or the left side. Since 24 years old man is not on the right side, and we only have 20 or 24 years old people, so 20 years old person is on the right side. In other words, we got 20-year-old accountant.

since at least one of the two people sitting on the right side of 24 years old, thus the left side is 24 years old, so we have 24-year-old salesperson. Since we must have 2 people siting on the right side of the 24 years old person and we only have 20 or 24 years old people, so only we have

left(24)     middle(20)      right(20)
salesperson, salesperson, accountant

Analyze the option:
(A) the option does not contrast to our inference results.
(B) the option does contrast to our inference results, since we should only have 1 24 year old person.
(C) the option contrast to our inference result, since we must have one 24-year-old salesperson.
(D) the option contrast to our inference result, since we only have one accuntant.


Therefore, the answer is (A)

Final Answer: A"""


    if 'MBPP' in task_name.upper():
        q1 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. \n\n\nTask: Write a function to find the longest chain which can be formed from the given set of pairs.\nTest Example: assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3\n\n1. We wish you to answer the question.\n2. You must directly provide the code answer without say anything else. Please not saying anything 'like sure I can help you with'.\n3. The code should be runnable code which means You do not need to add ``` ``` or add python in front of the code.\n4. The test is only used to show you the input structure. Please do not run the test in your answer.\n"""
        
        q2 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. \n\n\nTask: Write a python function to find the first repeated character in a given string.\nTest Example: assert first_repeated_char("abcabc") == "a"\n\n1. We wish you to answer the question.\n2. You must directly provide the code answer without say anything else. Please not saying anything \'like sure I can help you with\'.\n3. The code should be runnable code which means You do not need to add ``` ``` or add python in front of the code.\n4. The test is only used to show you the input structure. Please do not run the test in your answer.\n"""
        
        gold_label1 = \
"""class Pair(object): \r\n\tdef __init__(self, a, b): \r\n\t\tself.a = a \r\n\t\tself.b = b \r\ndef max_chain_length(arr, n): \r\n\tmax = 0\r\n\tmcl = [1 for i in range(n)] \r\n\tfor i in range(1, n): \r\n\t\tfor j in range(0, i): \r\n\t\t\tif (arr[i].a > arr[j].b and\r\n\t\t\t\tmcl[i] < mcl[j] + 1): \r\n\t\t\t\tmcl[i] = mcl[j] + 1\r\n\tfor i in range(n): \r\n\t\tif (max < mcl[i]): \r\n\t\t\tmax = mcl[i] \r\n\treturn max"""
        gold_label2 = \
"""def first_repeated_char(str1):\r\n  for index,c in enumerate(str1):\r\n    if str1[:index+1].count(c) > 1:\r\n      return c \r\n  return \"None\""""
        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""from dataclasses import dataclass\n\n@dataclass\nclass Pair:\n    first: int\n    second: int\n\ndef max_chain_length(pairs, n):\n    # Sort the pairs based on the second element\n    pairs.sort(key=lambda x: x.second)\n\n    # Initialize the maximum length of chain\n    max_length = 1\n    last_in_chain = pairs[0]\n\n    # Loop through the sorted pairs and find the maximum chain length\n    for i in range(1, n):\n        if pairs[i].first > last_in_chain.second:\n            max_length += 1\n            last_in_chain = pairs[i]\n\n    return max_length"""
            a2 = \
"""def first_repeated_char(input_string):\n    seen = set()\n    for char in input_string:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None  # Return None if there's no repeated character"""

        elif varient_name == 'openai_human_written_examples':
            a1 = \
"""class Pair(object):
    # Constructor to initialize a pair with two values (a, b)
    def __init__(self, a, b):
        self.a = a
        self.b = b

def max_chain_length(arr, n):
    # Initialize max length of chain
    max = 0

    # Create an array to store the maximum chain length at each index
    mcl = [1 for i in range(n)]

    # Compute the maximum chain length for each pair
    for i in range(1, n):
        for j in range(0, i):
            # If the current pair can be chained to the previous one
            if (arr[i].a > arr[j].b and mcl[i] < mcl[j] + 1):
                # Update the maximum chain length for the current pair
                mcl[i] = mcl[j] + 1

    # Find the maximum value in mcl (maximum chain length)
    for i in range(n):
        if (max < mcl[i]):
            max = mcl[i]

    return max"""

            a2 = \
"""def first_repeated_char(str1):
    # Function to find the first repeated character in a given string.

    # Parameters:
    # str1 (str): Input string to check for repeated characters.

    # Returns:
    # str: The first repeated character, or "None" if no character repeats.

    # Iterate through each character in the string along with its index
    for index, c in enumerate(str1):
        # Check if the character has appeared more than once in the substring up to the current index
        if str1[:index + 1].count(c) > 1:
            return c  # Return the first repeated character
    
    return "None"  # Return "None" if no repeated character is found"""


    if 'API_BANK' in task_name.upper():
        q1 = \
"""\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request:\n\nPlease inference first, then generate the API-Request at the end after 'Final Answer:'"""
                
        q2 = \
"""\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nGenerate next API Request:\n\nPlease inference first, then generate the API-Request at the end after 'Final Answer:'"""

        q3 = \
"""\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]->{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\nGenerate next API Request:\n\nPlease inference first, then generate the API-Request at the end after 'Final Answer:'"""
        gold_label1 = "Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]"
        gold_label2 = "Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]"
        gold_label3 = "Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]"

        if varient_name == 'gpt4_style_in_context_examples':
            a1 = \
"""INFERENCE: The user's request is to find a cardiologist in Los Angeles for a check-up appointment. Based on this request, we need to search for a tool that can help with healthcare provider search and appointment availability in Los Angeles. The best matching keyword for this search from the available APIs would be \"healthcare provider appointment availability checker\", as it seems to cover both the search for a healthcare provider (in this case, a cardiologist) and checking for appointment availability.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]"""
            a2 = \
"""INFERENCE: To generate the correct API request based on the user's utterance, we first need to understand what the user is asking for. The user is looking to find a cardiologist in Los Angeles for a check-up appointment. This implies the need for checking healthcare provider availability in a specific specialty and location. Based on the available APIs, the \"HealthcareProviderAppointmentChecker\" matches the requirement to check for healthcare providers' appointments. The parameters required are 'specialty', which will be 'cardiologist', and 'location', which will be 'Los Angeles'.\n\nFinal Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]"""
            a3 = \
"""The user's request is to find a cardiologist in Los Angeles for a check-up appointment. This scenario involves two primary tasks: identifying the appropriate API to search for healthcare-related tools and using a specialized API to find appointment availability with a healthcare provider.\n\nThe first step is to utilize the `ToolSearcher` API to identify a suitable tool which can handle appointment searching and scheduling. Based on the description of the `ToolSearcher` API, it searches for relevant tools in a library using keywords. The keyword in this case would relate to healthcare appointments.\n\nThe next step, after identifying the tool, would typically involve using it to find the required appointments with a cardiologist. But before reaching that, we focus on finding an appropriate scheduler tool.\n\nGiven these steps and the user's goal, I will form a request to the `ToolSearcher` API using the keyword that captures the need for a healthcare appointment scheduler.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]"""

        if varient_name == 'openai_human_written_examples':
#             a1 = \
# """The user is requesting to find a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The first step should be to search for a tool that can help find a healthcare provider appointment availability checker. To accomplish this, we choose the ToolSearcher API from the available APIs. The ToolSearcher API is used to search for relevant tools based on the provided keywords according to the description. We need to fill out the keywords according to the description. The keywords could be 'healthcare provider appointment availability checker.' Therefore, the next step (which is also the first step) is:

# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]"""
#             a2 = \
# """According to the API call history, the user has called the ToolSearcher API and found the HealthcareProviderAppointmentChecker API. The next step is to fill out the input parameters for HealthcareProviderAppointmentChecker and use it to find healthcare provider appointment availability. The input parameters are specialty and location. The user wants to find a cardiologist in Los Angeles for a check-up appointment. Therefore, the next API request should be:

# Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]"""
#             a3 = \
# """The user previously called the HealthcareProviderAppointmentChecker API and found three appointment times, which are '2034-04-18 14:30:00', '2034-04-19 11:00:00', and '2034-04-20 09:45:00'. The next step is to find the scheduler for the appointment. Since there is no available tool, the user needs to search for a tool that can schedule healthcare provider appointments. The ToolSearcher API can be used to search for relevant tools based on the keywords according to the description. The keywords should be 'healthcare provider appointment scheduler'. Therefore, the answer is:

# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]"""
#             a1 = \
# """The user is requesting to find a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The next step should be to search for a tool that can help find a healthcare provider appointment availability checker. To accomplish this, we choose the ToolSearcher API from the available APIs. The ToolSearcher API is used to search for relevant tools based on the provided keywords according to the description. We need to fill out the keywords according to the description. The keywords could be 'healthcare provider appointment availability checker.' Therefore, the next step is:

# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]"""
#             a2 = \
# """According to the API history, the user has already called the ToolSearcher API and found the HealthcareProviderAppointmentChecker API. Tehrefore, the user just need to to fill out the input parameters for HealthcareProviderAppointmentChecker and use it to find healthcare provider appointment availability. The input parameters for HealthcareProviderAppointmentChecker are specialty and location. Since the user wants to find a cardiologist in Los Angeles for a check-up appointment, we can set the input parameter for specialty as 'cardiologist' and set the input parameter for location as 'Los Angeles'. Therefore, the next API request should be:

# Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]"""
            a1 = \
"""the user is looking for a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The api that could help the user is ToolSearcher. the api that could use ToolSearcher api for find a tool to check if there are any healthcare provider who can provide appointment at the time. We will need to fill out the input parameters for ToolSearcher api according to the description. according to the description, we need to fill out the keyword in string format to search for and it will return us an available tool. the user is looking for a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment thus the key input in this case would be 'healthcare provider appointment availability checker'. we need to use ToolSearcher to find a checker for healthcare provider appointment booking. According to the question, we need to generate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)]. Therefore, our final API-CALL is

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]"""
            a2 = \
"""The user is looking for a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. According to the available api, the one matches user's need is HealthcareProviderAppointmentChecker. According to the description from HealthcareProviderAppointmentChecker, the user could use HealthcareProviderAppointmentChecker to checking the availability of appointments with healthcare providers. According to the description, we need to fill out the keyword \'specialty\' in string format. According to the description of the input, the input is for determine the the specialty of the healthcare. According to user's inquary, he is looking for a healthcare provider which specifically a cardiologist. Thus the input parameter for \'specialty\' is 'cardiologist'. Another input for HealthcareProviderAppointmentChecker is location. Accoding to user's inquary, he is looking for a health provider in Los Angeles, thus we need to set location to Los Angeles. According to the question, we need to generate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)]. Therefore, our final API-CALL is

Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]"""
            a3 = \
"""The user is looking for a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. According to the previous api calling history, the user has already use HealthcareProviderAppointmentChecker to find the cardiologist located in Los Aneles. According to the api call history, this api call will return user a list of available appointment slots. However, with the slots, what would the user to do next? an user would tipically read over the slots and determine which time is best for himself. After that, he would need to book the appointment with the health provider. However, there is no scheduler available in the api list. Therefore, the user would need to use ToolSearch API from the api list to look for a scheduler api tool for schedule a time with the health provider. The ToolSearch API requires the user to input the keywords in string. The user would put 'healthcare provider appointment scheduler' as input string for search for scheduler api. According to the question, we need to generate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)]. Therefore, our final API-CALL is

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]"""

    if task_name == 'ESNLI' or task_name == 'API_BANK':
#     if task_name == 'ESNLI':
        prompt = f"""We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : "{q1}"

groundtruth: {gold_label1}

Inference: {a1}




Example 2:

question: "{q2}"

groundtruth: {gold_label2}

Inference: {a2}



Example 3:

question: "{q3}"

groundtruth: {gold_label3}

Inference: {a3}


We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""
    elif task_name == 'MBPP':
        prompt = f"""We have the "{question}" and the groundtruth {gold_label}


1. We wish you to transform the groundtruth to the example style. We will use your transformed groundtruth to train our model, thus you will transform and pretend as not knowing the gold_label.
2. You must make the groundtruth transformation directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'

Example 1:

Question : "{q1}"

groundtruth: {gold_label1}

transformaed groundtruth: {a1}




Example 2:

question: "{q2}"

groundtruth: {gold_label2}

transformed groundtruth: {a2}




We have the "{question}" and the groundtruth {gold_label}


transformaed groundtruth: 
Please make the transformation directly without mention things like 'sure i can help you' or 'transformed groundtruth: '
"""
    else:
        prompt = f"""We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : "{q1}"

groundtruth: {gold_label1}

Inference: {a1}




Example 2:

question: "{q2}"

groundtruth: {gold_label2}

Inference: {a2}




We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""
    return prompt



def provide_self_generated_example_template(gold_label, question, template_holder):
    a1 = template_holder['answer1']
    a2 = template_holder['answer2']
    gold_label1 = template_holder['gold_label1']
    gold_label2 = template_holder['gold_label2']
    q1 = template_holder['question1']
    q2 = template_holder['question2']

#     print('---------------------------------------------------------------------------')
#     print('example1: ', a1)
#     print()
#     print('///////////////////////////////////////////////////////////////////////////')
#     print()
#     print('example2: ', a2)

#     prompt = f"""We have the "{question}" and the groundtruth {gold_label}


# 1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

# INFERENCE HERE
# Final Answer: {gold_label}

# Example 1:

# Question : "{q1}"

# groundtruth: {gold_label1}

# Inference: {a1}




# Example 2:

# question: "{q2}"

# groundtruth: {gold_label2}

# Inference: {a2}




# We have the "{question}" and the groundtruth {gold_label}


# 1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

# INFERENCE HERE
# Final Answer: {gold_label}"""

#     prompt = f"""We have the "{question}" and the groundtruth {gold_label}


# 1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

# INFERENCE HERE
# Final Answer: {gold_label}

# Example:

# Question: "{q1}"

# groundtruth: {gold_label1}

# Inference: {a1}




# We have the "{question}" and the groundtruth {gold_label}


# 1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

# INFERENCE HERE
# Final Answer: {gold_label}"""

    prompt = f"""We have the "{question}", the groundtruth {gold_label} and 2 examples.

Based on the question and the groundtruth, we wish you to create a better groundtruth for training. The better groundtruth has to have similar style as the 2 examples. The 2 examples sometimes contain errors. It is ok. we just wish to show you the style.
1. We wish you to re-answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
3. You will provide better groundtruth first then put the Final Answer: {gold_label} at the end of the prediction like this

BETTER_GROUNDTRUTH_HERE
Final Answer: GOLD_LABEL_HERE



Example:

Inference Question: "{q1}"

Inference Groundtruth: {gold_label1}

Example2 Inference: {a1}




Example2 Question: "{q2}"

Example2 Groundtruth: {gold_label2}

Example2 Inference: {a2}


Please use the similar styles as shown in the examples above. Remember to attach Final Answer: GOLD_LABEL_HERE at the end like this

BETTER_GROUNDTRUTH_HERE
Final Answer: {gold_label}"""
    return prompt

# Here we will provide you two examples, which sometimes contain errors. It is ok. we just wish to show you the style.
def provide_self_generated_example_template_plan_bench_or_api_bank(gold_label, question, template_holder):
    a1 = template_holder['answer1']
    a2 = template_holder['answer2']

#     prompt = f"""We have the "{question}", the groundtruth {gold_label} and 2 examples.

# Based on the question and the groundtruth, we wish you to create a better groundtruth for training. The better groundtruth has to have similar style as the 2 examples. The 2 examples sometimes contain errors. It is ok. we just wish to show you the style.
# 1. We wish you to re-answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

# INFERENCE_HERE
# Final Answer: GOLD_LABEL_HERE



# example 1: {a1}

# example 2: {a2}


# Please use the similar styles as shown in the inference process examples above. Remember to attach Final Answer: GOLD_LABEL_HERE at the end like this



# INFERENCE_HERE
# Final Answer: {gold_label}"""
    prompt = f"""We have the "{question}", the groundtruth {gold_label} and 2 examples.

Based on the question and the groundtruth, we wish you to create a better groundtruth for training. The better groundtruth has to have similar style as the 2 examples. The 2 examples sometimes contain errors. It is ok. we just wish to show you the style.
1. We wish you to re-answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
3. You will provide better groundtruth first then put the Final Answer: {gold_label} at the end of the prediction like this

BETTER_GROUNDTRUTH_HERE
Final Answer: GOLD_LABEL_HERE



example 1: {a1}

example 2: {a2}


Please use the similar styles as shown in the examples above. Remember to attach Final Answer: GOLD_LABEL_HERE at the end like this

BETTER_GROUNDTRUTH_HERE
Final Answer: {gold_label}"""
    return prompt


