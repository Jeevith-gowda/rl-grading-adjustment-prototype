# Sample submissions

Ten representative rows from the filtered dataset (question 9 excluded). 
Use these to sanity-check how AI scores, TA grades, and feedback align in practice.

## Sample 1: Exact agreement

- **submission_id:** 20
- **questions_id:** 3
- **Assignment:** Arrange the following functions in order of growth, from lowest to highest (asymptotic order):_x000d_
- **ai_score:** 8.0
- **TA_Grade:** 8.0
- **Adjustment (TA − AI):** +0.0

### Student response

```
nÂ²n < 5 lg (n+1000)â´ < 3000000 nÂ³ < 0.001nâ´ + 3nÂ³ + 1 < 3â¿ < 2â¿ < (n-1)!
```

### AI feedback

```
The student did a good job in arranging the functions in order, but made a minor mistake. The ordering of 3^n and 2^(2n) should be swapped as 2^2n grows faster than 3^n. Moreover, the explanation of the notation ln^2n is off, as it implies square of logarithm of n, not n times squared logarithm of n, which is n*(lnn)^2. These two errors led to a deduction of 2 points from total score: 1.5 points from "Correct Ordering of Functions" and 0.5 points from "Justification and Reasoning". Keep refin...
```

**Interpretation:** AI and TA agreed on the score — no adjustment needed.

---

## Sample 2: Exact agreement

- **submission_id:** 34
- **questions_id:** 3
- **Assignment:** Arrange the following functions in order of growth, from lowest to highest (asymptotic order):_x000d_
- **ai_score:** 8.0
- **TA_Grade:** 8.0
- **Adjustment (TA − AI):** +0.0

### Student response

```
The function in order of growth from lowest to highest:

5lg(n+1000)â´ < lnÂ²n < 300000nÂ³ < 0.001nâ´+3nÂ³+1 < 3â¿ < (n-1)!
```

### AI feedback

```
Correct Ordering of Functions: The student correctly ordered most of the functions. However, the student made a mistake in ordering the last two terms. 2â¿ grows slower than (n-1)! as the base is smaller in a direct comparison, resulting in 3.5/4 for this section.

Justification and Reasoning: The reasoning given by the student is generally correct, explaining the difference in growth rates using Big-O notation and highlighting the distinction between logarithmic, linear, polynomial, exponen...
```

**Interpretation:** AI and TA agreed on the score — no adjustment needed.

---

## Sample 3: AI underscored by 2–3 points

- **submission_id:** 378
- **questions_id:** 7
- **Assignment:** 1)Estimate how many times faster quicksort will sort an array of one million random numbers on average than selection...
- **ai_score:** 8.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +2.0

### Student response

```
Quicksort  
Tuesday, September 23, 2025 4:37 PM

[2 points] Estimate how many times faster quicksort will sort an array of one million random numbers on average than selection sort.

Note that in your answer, please include â€œQuicksort is roughlyÂ 50,000Â Ã— faster than selection sortÂ for one million numbers.â€

\[ \text{n} = 1000000 \]

\(\text{Quicksort: }  O(n \log n)\) 

\[ n \cdot \log n = 1000,000 \cdot \log 1,000,000 = 1000,000 \cdot 20  \]

\(\text{Selection sort: }  O(n^{2})\) 

\...
```

### AI feedback

```
You correctly stated that quicksort is roughly 50,000 times faster than selection sort and identified that E is the pivot. You also correctly noted that insertion sort can beat quicksort on sorted arrays. However, you incorrectly stated the final array state at the end of the second pass; it should be [E, A, X, M, P, L, E] instead of [E, E, A, M, P, L, X]. To improve, ensure the final array state reflects the correct swaps made during the partitioning process.
```

**Interpretation:** AI undergraded by 2.0 points — typical pattern on Q7 Quicksort where the TA bumps scores for reasonable but imperfect answers.

---

## Sample 4: AI underscored by 2–3 points

- **submission_id:** 183
- **questions_id:** 4
- **Assignment:** Consider the following two algorithms for finding whether an array A[1..n] contains a pair of elements whose sum equa...
- **ai_score:** 8.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +2.0

### Student response

```
Time Complexity Analysis for Non-recursive Algorithms

Algorithm #1:
Base Operation: when a valid pair is found in the first comparison, let i = 0, j = 1  then number of comparison = 1. Time complexity = O(n)
Worst Case: No such pair exists, loops: i = 0 to n-1, j = i+1 to n-1 then number of comparison: âˆ‘ from i=0 to n-1 âˆ‘ from j=i+1 to n-1 1 = âˆ‘ from i=0 to n-2 (n-i-1) = n(n-1)/2 = O(n^2)
Best case: when a valid pair is found in the first comparison. when i = 0, j = 1 then number of co...
```

### AI feedback

```
- Understanding of Time Complexity: Good understanding demonstrated. (3/3)
- Analysis of Algorithm 1: Well analyzed and explained. (2/2)
- Analysis of Algorithm 2: Your analysis has a slight error. The worst case for insertion sort is O(n^2), and the two pointer search has a worst-case time complexity of O(n). As such, the total worst case time complexity considering both components should be O(n^2) + O(n), not O(n^2) + O(n^2). Best case is also misinterpreted. (2/3)
- Comparison & Conclusion...
```

**Interpretation:** AI undergraded by 2.0 points — typical pattern on Q4 Non-recursive analysis where the TA bumps scores for reasonable but imperfect answers.

---

## Sample 5: AI underscored by 2–3 points

- **submission_id:** 573
- **questions_id:** 8
- **Assignment:** Scenario:_x000d_
- **ai_score:** 8.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +2.0

### Student response

```
1) Heap Vs Arrays

A binary heap supports insertion and min extraction
in O(logn), versus O(n) worst case for insertion,
and min extraction in a sorted or unsorted array.

Reason
* Binary heap supports insertion O(logn) and min
  extraction O(logn)
* Sorted array: insertion is O(n), min extraction is O(1)
* Unsorted array: insertion is O(1), min extraction is O(n)

2) Array representation

We insert elements one by one.
1. Insert 45 â†’ [45]
2. Insert 20 â†’ [20, 45]
3. Insert 60 â†’ [20, 45,...
```

### AI feedback

```
Great job on explaining the efficiency of binary heaps compared to arrays and providing a clear array representation of the final min-heap! However, there are some inaccuracies in the index relationships for the node with priority 15; its parent index should be 2 and value 20, not 1 and 10. Additionally, the final min-heap after removal should be [15, 20, 35, 45, 50, 60] instead of [15, 20, 50, 45, 35, 60]. Please double-check these details for accuracy. Keep up the good work!
```

**Interpretation:** AI undergraded by 2.0 points — typical pattern on Q8 Heap where the TA bumps scores for reasonable but imperfect answers.

---

## Sample 6: AI underscored by 5+ points

- **submission_id:** 290
- **questions_id:** 5
- **Assignment:** In this exercise, you will practice applying the substitution method to analyze the asymptotic order of running times...
- **ai_score:** 0.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +10.0

### Student response

```
The student's answer is only the solution to the first task but it is not correct. They have used a linear recurrence relation instead of a recursive relation that describes the given frying hamburger problem.
```

### AI feedback

```
The complexity analysis for the given problem of frying hamburgers is incorrect. The provided algorithm is not representative of the problem complexity, which is a task-dependent recurrence relation. The correct recurrence relation should have taken into account that two hamburgers can be fried at the same time. Additionally, tasks 2 and 3 are not answered. Please review the assignment instructions carefully and ensure you understand the problem statement before starting your solution. Specif...
```

**Interpretation:** AI undergraded by 10.0 points — large gap on Q5 Substitution, often seen on algorithm-heavy items where the TA restores partial credit.

---

## Sample 7: AI underscored by 5+ points

- **submission_id:** 313
- **questions_id:** 6
- **Assignment:** In this exercise, you will practice using the Master Method to analyze the asymptotic running time of recursive algor...
- **ai_score:** 0.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +10.0

### Student response

```
The student did not provide an answer.
```

### AI feedback

```
Unfortunately, the student did not respond to the question and therefore did not demonstrate understanding of the Master Method, nor show an ability to apply it to solve recurrence relations. The student needs to follow the prompt instructions in order to get any points. I recommend the student review the Master Method and consider attempting this question again.
```

**Interpretation:** AI undergraded by 10.0 points — large gap on Q6 Master theorem, often seen on algorithm-heavy items where the TA restores partial credit.

---

## Sample 8: AI overscored (TA lowered grade)

- **submission_id:** 1653
- **questions_id:** 10
- **Assignment:** This lab is designed to give students practical experience with Dynamic Programming by comparing different solutions...
- **ai_score:** 6.0
- **TA_Grade:** 0.0
- **Adjustment (TA − AI):** -6.0

### Student response

```
Sketchbook: 0-1 Knapsack and the Dynamic Programming_x000d_
MOHAN KRISHNA OTIKUNTA_x000d_
801418781_x000d_
 1. In the recursive version of the Knapsack problem, one of the earliest paths where the function first reaches n=1 will result in the invocation KnapsackRecursive(values, weights, 1, 0) _x000d_
_x000d_
2. In the Knapsack problemâ€™s top-down dynamic programming solution, the cell dp[2][50] represents the maximum value that can be achieved by selecting from the first 2 items to fit into...
```

### AI feedback

```
You correctly identified the recursive invocation and the meaning of the dp[2][50] cell in the top-down approach, earning points for those. However, the second function invocation in the top-down solution is incorrect, as it should be KnapsackTopDown(values, weights, 2, 20, dp). Additionally, you did not provide the correct value for dp[2][20], which is crucial for full credit. Please ensure to double-check these details in future responses.
```

**Interpretation:** AI overgraded by 6.0 points — TA lowered the score; may reflect rubric nuance on Q10 0-1 Knapsack.

---

## Sample 9: Easier question (Q3)

- **submission_id:** 13
- **questions_id:** 3
- **Assignment:** Arrange the following functions in order of growth, from lowest to highest (asymptotic order):_x000d_
- **ai_score:** 9.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +1.0

### Student response

```
5lg(n+1000)^4 < ln^2n < 3000000n^3 < 0.0001n^4.3n^3+1 < 3^n < 2^2^n < (n-1)!
```

### AI feedback

```
- Correct Ordering of Functions (4/4): You've perfectly arranged the functions in order from lowest to highest growth. 
- Justification and Reasoning (2.5/3): You provided clear reasoning for most comparisons, but there was a minor error in reasoning for logarithmic functions. You used L'Hospitals rule incorrectly. For correct ordering, you should've simply noted that the exponent of the logarithm changes the constant factor, not the growth rate.
- Clarity and Communication (1.5/2): Your expl...
```

**Interpretation:** AI undergraded by 1.0 points on Q3 Order notation — TA applied a modest upward adjustment.

---

## Sample 10: Harder question (Q11)

- **submission_id:** 1246
- **questions_id:** 11
- **Assignment:** This mini-lab guides students through solving the fractional knapsack problem using the greedy algorithm, emphasizing...
- **ai_score:** 0.0
- **TA_Grade:** 10.0
- **Adjustment (TA − AI):** +10.0

### Student response

```
nan
```

### AI feedback

```
It seems that there is no response provided to evaluate. Please ensure to include your answers to the questions in order to receive a score.
```

**Interpretation:** AI undergraded by 10.0 points — large gap on Q11 Fractional Knapsack, often seen on algorithm-heavy items where the TA restores partial credit.

---
